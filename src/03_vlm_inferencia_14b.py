"""
Inferencia offline con InternVL3-14B usando vLLM.

Diferencias clave respecto al script genérico (03_vlm_inferencia_local.py):
  - Las reglas anti-alucinación se colocan en el SYSTEM message (se envían una
    sola vez, no repiten en cada prompt de usuario), lo que reduce tokens y
    es más efectivo en modelos de chat de tamaño medio.
  - El prompt de usuario es deliberadamente más conciso: InternVL3-14B rinde
    mejor con instrucciones focalizadas que con bloques de texto muy largos.
  - trust_remote_code=True es obligatorio para cargar los archivos de modelo
    custom (modeling_internvl_chat.py, configuration_internvl_chat.py).
  - Defaults ajustados para 2× A100-80GB (--tp 2).
"""

import re
import json
import logging
import argparse
from pathlib import Path

from vllm import LLM, SamplingParams

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("internvl14b_offline")

# ---------------------------------------------------------------------------
# SYSTEM PROMPT — enviado una sola vez por request, no en cada user message
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a specialized biomedical figure analysis assistant. "
    "You extract information strictly from what is visually present in the image pixels.\n\n"
    "CRITICAL DATA INTEGRITY RULES:\n"
    "1. Numeric values, axis labels, scale bars, band positions — your ONLY source is the IMAGE PIXELS.\n"
    "2. NEVER copy, estimate, or infer numbers from the abstract, RAG passages, or caption.\n"
    "3. If a value is not explicitly printed and clearly readable in the pixels, write exactly: "
    '"Values not explicitly written"\n'
    "4. The abstract, RAG, and caption are provided for INTERPRETATION only — never as a data source.\n"
    "5. Respond ONLY with a valid JSON object. No markdown fences. No preamble. No trailing text."
)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
PROMPT_ROUTER = (
    "Classify this scientific image into exactly one of the following categories.\n"
    "Respond ONLY with the category name, nothing else.\n"
    "Categories: [Data Chart, Microscopy, Pathway/Flowchart, Gel/Blot, Conceptual Model, Table]"
)

PROMPTS_EXPERTOS = {
    "Data Chart": (
        "Analyze this data chart. Describe all visible panels. "
        "For each panel identify: plot type, axes (labels + units), legend entries, "
        "statistical markers (*, **, p-values, error bars), and dominant visual trends."
    ),
    "Gel/Blot": (
        "Analyze this gel/blot. Identify: number of lanes and their labels, "
        "molecular weight markers and their values (if printed), "
        "loading controls (Actin, GAPDH, etc.), and relative band intensities."
    ),
    "Pathway/Flowchart": (
        "Trace this biological pathway or process flow. Identify: "
        "receptors/entry points, key nodes, arrow directions, "
        "activation (arrow) vs. inhibition (flat-head) relationships, "
        "and any color-coded states or compartments."
    ),
    "Microscopía": (
        "Analyze this microscopy image. Identify: scale bar value and units (if printed), "
        "fluorescent channels and their colors, cellular or subcellular structures visible, "
        "and any phenotypic differences between panels."
    ),
    "Table": (
        "Extract the data from this table. Identify: column headers, row labels, "
        "units of measurement, and the key comparative findings across rows/columns."
    ),
    "Conceptual Model": (
        "Describe this conceptual model or diagram. Identify: main components, "
        "labeled nodes or compartments, directional relationships between them, "
        "and the biological hypothesis or mechanism being illustrated."
    ),
}

JSON_SCHEMA = """{
  "visual_content": "<objective pixel-level description: panel layout, shapes, colors, axes, all visible text>",
  "data_observed": "<ONLY values and labels explicitly printed and legible in the pixels. Use \\"Values not explicitly written\\" for anything not directly visible>",
  "interpretation": "<synthesis integrating the visual content with the scientific context provided>"
}"""

# Presupuesto de caracteres para el contexto textual del prompt
_RAG_CHAR_LIMIT = 3000
_ABSTRACT_CHAR_LIMIT = 2000
_CAPTION_CHAR_LIMIT = 600


def _truncate(text: str, char_limit: int, label: str) -> str:
    if len(text) > char_limit:
        logger.warning(f"Truncando {label}: {len(text)} → {char_limit} chars")
        return text[:char_limit] + " [TRUNCATED]"
    return text


def clean_json_output(raw_text: str) -> dict:
    """Extrae el primer bloque JSON válido ignorando markdown o texto extra."""
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
    if match:
        candidate = match.group(1)
    else:
        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return {"parse_error": "No JSON object found", "raw_output": raw_text}
        candidate = raw_text[start : end + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as e:
        return {"parse_error": str(e), "raw_output": raw_text}


def build_multimodal_message(system: str, user_text: str, image_path: Path) -> list:
    """Construye el mensaje de chat con system prompt + contenido multimodal."""
    # file:// permite a vLLM cargar la imagen directamente del disco (C++),
    # evitando codificación base64 y reduciendo uso de RAM.
    abs_path = image_path.resolve()
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": f"file://{abs_path}"}},
            ],
        },
    ]


def build_router_message(image_path: Path) -> list:
    """El router no necesita system prompt complejo."""
    abs_path = image_path.resolve()
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT_ROUTER},
                {"type": "image_url", "image_url": {"url": f"file://{abs_path}"}},
            ],
        }
    ]


def process_interpretations_offline(
    context_json: Path,
    model_path: str,
    tp_size: int = 2,
    max_model_len: int = 16384,
    output_name: str = "interpretations_14b.json",
):
    stem = context_json.parent.name
    out_file = context_json.parent / output_name

    with open(context_json, "r", encoding="utf-8") as f:
        items = json.load(f)

    logger.info(f"[{stem}] Cargando InternVL3-14B desde: {model_path}")
    logger.info(f"TP={tp_size} | max_model_len={max_model_len} | items={len(items)}")

    llm = LLM(
        model=model_path,
        tensor_parallel_size=tp_size,
        trust_remote_code=True,         # obligatorio para archivos custom de InternVL
        max_model_len=max_model_len,
        gpu_memory_utilization=0.92,
        limit_mm_per_prompt={"image": 1},
        allowed_local_media_path=str(context_json.parent.resolve()),
    )

    # -----------------------------------------------------------------------
    # PASO 1 — Clasificador en batch
    # -----------------------------------------------------------------------
    logger.info(f"[{stem}] 1/2: Clasificando {len(items)} imágenes en batch...")

    router_messages = []
    valid_items = []

    for item in items:
        img_path = context_json.parent / item["crop_path"]
        if img_path.exists():
            router_messages.append(build_router_message(img_path))
            valid_items.append((item, img_path))
        else:
            logger.warning(f"Imagen no encontrada: {img_path}")

    router_params = SamplingParams(temperature=0.0, max_tokens=20)
    router_outputs = llm.chat(messages=router_messages, sampling_params=router_params)

    categorias_detectadas = []
    for out in router_outputs:
        cat_cruda = out.outputs[0].text.strip()
        cat_final = "Conceptual Model"  # fallback
        for cat in PROMPTS_EXPERTOS:
            if cat.lower() in cat_cruda.lower():
                cat_final = cat
                break
        categorias_detectadas.append(cat_final)

    logger.info(
        f"[{stem}] Categorías detectadas: "
        + ", ".join(f"{c}={categorias_detectadas.count(c)}" for c in set(categorias_detectadas))
    )

    # -----------------------------------------------------------------------
    # PASO 2 — Intérprete especializado en batch
    # -----------------------------------------------------------------------
    logger.info(f"[{stem}] 2/2: Análisis especializado en batch...")

    expert_messages = []
    for (item, img_path), categoria in zip(valid_items, categorias_detectadas):
        prompt_base = PROMPTS_EXPERTOS[categoria]

        rag_context = _truncate(
            "\n\n".join(item["bm25_top_passages"]), _RAG_CHAR_LIMIT, f"RAG [{item['item_id']}]"
        )
        abstract = _truncate(
            item["abstract_paper"], _ABSTRACT_CHAR_LIMIT, f"abstract [{item['item_id']}]"
        )
        caption = _truncate(
            item["caption_oficial"], _CAPTION_CHAR_LIMIT, f"caption [{item['item_id']}]"
        )

        user_text = f"""{prompt_base}

=== PAPER ABSTRACT ===
{abstract}

=== FIGURE CONTEXT (BM25-RAG) ===
{rag_context}

=== OFFICIAL CAPTION ===
{caption}

Respond with this JSON structure (no markdown, no extra text):
{JSON_SCHEMA}"""

        expert_messages.append(build_multimodal_message(SYSTEM_PROMPT, user_text, img_path))

    # temperature=0.0 para máximo determinismo; max_tokens=3000 evita truncamiento
    expert_params = SamplingParams(temperature=0.0, max_tokens=3000)
    expert_outputs = llm.chat(messages=expert_messages, sampling_params=expert_params)

    # -----------------------------------------------------------------------
    # Ensamblar y guardar
    # -----------------------------------------------------------------------
    resultados = []
    parse_errors = 0
    for (item, _), categoria, out in zip(valid_items, categorias_detectadas, expert_outputs):
        raw_text = out.outputs[0].text.strip()
        parsed = clean_json_output(raw_text)
        if "parse_error" in parsed:
            parse_errors += 1
            logger.warning(f"Parse error en {item['item_id']}: {parsed['parse_error']}")
        resultados.append({
            "item_id": item["item_id"],
            "categoria_visual": categoria,
            "interpretacion_llm": parsed,
        })

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)

    logger.info(
        f"[{stem}] Completado. {len(resultados)} items → {out_file} "
        f"| Parse errors: {parse_errors}/{len(resultados)}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Inferencia offline con InternVL3-14B (vLLM)"
    )
    parser.add_argument("--json_dir", required=True, help="Directorio con context.json")
    parser.add_argument(
        "--model_path",
        default="/workspace1/rubenqc/.ibio/chart-to-text/models/internvl3-14b",
        help="Ruta local al modelo InternVL3-14B",
    )
    parser.add_argument("--tp", type=int, default=2, help="Tensor Parallel (GPUs, default: 2)")
    parser.add_argument(
        "--max_model_len", type=int, default=16384,
        help="Contexto máximo en tokens (default: 16384)",
    )
    parser.add_argument(
        "--output_name", default="interpretations_14b.json",
        help="Nombre del archivo de salida (default: interpretations_14b.json)",
    )
    args = parser.parse_args()

    context_file = Path(args.json_dir) / "context.json"
    if not context_file.exists():
        logger.error(f"No se encontró context.json en {args.json_dir}")
        return

    process_interpretations_offline(
        context_file,
        args.model_path,
        args.tp,
        args.max_model_len,
        args.output_name,
    )


if __name__ == "__main__":
    main()
