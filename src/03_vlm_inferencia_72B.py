import re
import json
import logging
import argparse
from pathlib import Path

from vllm import LLM, SamplingParams

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("vlm_offline")

# --- Prompts ---
PROMPT_ROUTER = """Classify this scientific image into exactly one of the following categories.
Respond ONLY with the category name, nothing else.
Categories: [Data Chart, Microscopy, Pathway/Flowchart, Gel/Blot, Conceptual Model, Table]"""

PROMPTS_EXPERTOS = {
    "Data Chart": "Analyze this data chart. Identify the axes, legends, and statistical significance markers (*, p-values). Describe the key visual trends.",
    "Gel/Blot": "Analyze this gel/blot image. Identify molecular weight markers, distinct lanes, loading controls (e.g., Actin, GAPDH), and describe the relative expression levels of the bands.",
    "Pathway/Flowchart": "Trace the biological pathway or process flow shown. Identify key receptors, nodes, and directional relationships (activation/inhibition).",
    "Microscopía": "Analyze this microscopy image. Identify scale bars, cellular/subcellular structures, fluorescent markers or stains used, and describe the morphological phenotypes.",
    "Table": "Extract the structured data from this table. Identify the headers and units of measurement.",
    "Conceptual Model": "Describe the theoretical model or hypothesis illustrated. Identify the main components and how they interact.",
}


def clean_json_output(raw_text: str) -> dict:
    """Extrae el primer bloque JSON válido del texto del modelo, ignorando markdown."""
    # Intento 1: bloque ```json ... ``` o ``` ... ```
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
    if match:
        candidate = match.group(1)
    else:
        # Intento 2: primer '{' hasta el último '}' del texto
        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return {"parse_error": "No JSON object found", "raw_output": raw_text}
        candidate = raw_text[start : end + 1]

    try:
        return json.loads(candidate)
    except json.JSONDecodeError as e:
        return {"parse_error": str(e), "raw_output": raw_text}


def build_multimodal_message(prompt: str, image_path: Path) -> list:
    # file:// permite a vLLM cargar la imagen directamente del disco en C++,
    # evitando codificación base64 y reduciendo uso de RAM.
    abs_path = image_path.resolve()
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"file://{abs_path}"}},
            ],
        }
    ]


# Presupuesto de caracteres para el contexto textual del prompt experto.
# ~4 chars/token. Imagen (Qwen dyn-res) consume ~800-1500 tokens; reservamos
# el resto para texto + output (max_tokens=3000).
_RAG_CHAR_LIMIT = 3000
_ABSTRACT_CHAR_LIMIT = 2000
_CAPTION_CHAR_LIMIT = 600


def _truncate(text: str, char_limit: int, label: str) -> str:
    if len(text) > char_limit:
        logger.warning(f"Truncando {label} de {len(text)} a {char_limit} chars para caber en max_model_len")
        return text[:char_limit] + " [TRUNCATED]"
    return text


def process_interpretations_offline(
    context_json: Path, model_path: str, tp_size: int = 1, max_model_len: int = 16384
):
    stem = context_json.parent.name
    out_file = context_json.parent / "interpretations.json"

    with open(context_json, "r", encoding="utf-8") as f:
        items = json.load(f)

    logger.info(f"[{stem}] Cargando motor vLLM offline desde: {model_path}")
    logger.info(f"Tensor Parallel Size (GPUs): {tp_size} | max_model_len: {max_model_len}")

    llm = LLM(
        model=model_path,
        tensor_parallel_size=tp_size,
        trust_remote_code=True,   # necesario para InternVL; no-op para Qwen2.5-VL
max_model_len=max_model_len,
        gpu_memory_utilization=0.92,
        limit_mm_per_prompt={"image": 1},
        allowed_local_media_path=str(context_json.parent.resolve()),
    )

    # --- PASO 1: Enrutador en Batch (Clasificación) ---
    logger.info(f"[{stem}] 1/2: Clasificador visual en batch para {len(items)} imágenes...")

    router_messages = []
    valid_items = []

    for item in items:
        img_path = context_json.parent / item["crop_path"]
        if img_path.exists():
            router_messages.append(build_multimodal_message(PROMPT_ROUTER, img_path))
            valid_items.append((item, img_path))
        else:
            logger.warning(f"Imagen no encontrada: {img_path}")

    router_params = SamplingParams(temperature=0.0, max_tokens=20)
    router_outputs = llm.chat(messages=router_messages, sampling_params=router_params)

    categorias_detectadas = []
    for out in router_outputs:
        cat_cruda = out.outputs[0].text.strip()
        cat_final = "Conceptual Model"  # fallback
        for cat_experto in PROMPTS_EXPERTOS:
            if cat_experto.lower() in cat_cruda.lower():
                cat_final = cat_experto
                break
        categorias_detectadas.append(cat_final)

    # --- PASO 2: Intérprete Especializado en Batch ---
    logger.info(f"[{stem}] 2/2: Intérpretes especializados en batch...")

    expert_messages = []
    for (item, img_path), categoria in zip(valid_items, categorias_detectadas):
        prompt_base = PROMPTS_EXPERTOS[categoria]
        rag_context = _truncate(
            "\n\n".join(item["bm25_top_passages"]), _RAG_CHAR_LIMIT, f"RAG [{item['item_id']}]"
        )
        abstract = _truncate(item["abstract_paper"], _ABSTRACT_CHAR_LIMIT, f"abstract [{item['item_id']}]")
        caption = _truncate(item["caption_oficial"], _CAPTION_CHAR_LIMIT, f"caption [{item['item_id']}]")

        prompt_completo = f"""{prompt_base}

## CRITICAL ANTI-HALLUCINATION RULES — READ BEFORE ANSWERING

You are a strict scientific data extractor. Your ONLY source of truth for numerical values, axis labels, scale bars, and quantitative data is the PIXELS OF THE IMAGE ITSELF.

**MANDATORY RULES:**
1. DO NOT copy, estimate, or infer any numerical value from the text context.
2. DO NOT guess axis ranges, ticks, or values. Look carefully at the edges of bars, lines, and axes for small printed numbers.
3. If an image has multiple panels (a, b, c, d...), you MUST extract data strictly panel by panel, creating a key for EVERY visible panel.
4. If exact numbers are NOT printed next to data points or bars, you MUST output exactly: "Values not explicitly written".
5. IF a list of labels (e.g., y-axis accessions) has MORE THAN 10 items, DO NOT transcribe all of them. Instead, summarize the range and the number of items (e.g., "27 accession numbers from 6244 to 1741").

=== PAPER ABSTRACT ===
{abstract}

=== SPECIFIC FIGURE CONTEXT — RAG ===
{rag_context}

=== OFFICIAL CAPTION ===
{caption}

Respond ONLY with this JSON structure:
{{
  "visual_content": "Objective description of the layout, including all visible panels.",
  "data_observed": {{
      "panel_a": "Extracted data ONLY from panel A. Use 'Values not explicitly written' if needed.",
      "panel_b": "Extracted data ONLY from panel B. Include numbers printed at the ends of bars if visible.",
      "panel_c": "Extracted data ONLY from panel C.",
      "panel_d": "Extracted data ONLY from panel D. Include regression equations, R^2, and outlier labels."
  }},
  "interpretation": "Synthesis of the visual content with the context. DO NOT invent conclusions not explicitly supported by the text."
}}
"""
        expert_messages.append(build_multimodal_message(prompt_completo, img_path))

    # temperature=0.0 para máximo determinismo; max_tokens=3000 para evitar truncamiento
    expert_params = SamplingParams(temperature=0.0, max_tokens=3000)
    expert_outputs = llm.chat(messages=expert_messages, sampling_params=expert_params)

    # --- Ensamblar y Guardar Resultados ---
    resultados = []
    parse_errors = 0
    for (item, _), categoria, out in zip(valid_items, categorias_detectadas, expert_outputs):
        raw_text = out.outputs[0].text.strip()
        parsed = clean_json_output(raw_text)
        if "parse_error" in parsed:
            parse_errors += 1
            logger.warning(f"JSON parse error en {item['item_id']}: {parsed['parse_error']}")
        resultados.append({
            "item_id": item["item_id"],
            "categoria_visual": categoria,
            "interpretacion_llm": parsed,
        })

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)

    logger.info(
        f"[{stem}] Inferencia completada. {len(resultados)} items guardados en {out_file}. "
        f"Errores de parseo: {parse_errors}/{len(resultados)}"
    )


def main():
    parser = argparse.ArgumentParser(description="Etapa 3: Inferencia Local Offline con vLLM")
    parser.add_argument("--json_dir", required=True, help="Directorio que contiene el context.json")
    parser.add_argument("--model_path", required=True, help="Ruta local absoluta al modelo")
    parser.add_argument("--tp", type=int, default=1, help="Tensor Parallel Size (número de GPUs)")
    parser.add_argument(
        "--max_model_len", type=int, default=16384,
        help="Contexto máximo en tokens (default: 16384). Usar 8192 para modelos pequeños con poca VRAM."
    )
    args = parser.parse_args()

    context_file = Path(args.json_dir) / "context.json"
    if not context_file.exists():
        logger.error(f"No se encontró context.json en {args.json_dir}")
        return

    process_interpretations_offline(context_file, args.model_path, args.tp, args.max_model_len)


if __name__ == "__main__":
    main()
