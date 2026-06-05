import os
import re
import json
import logging
import argparse
from pathlib import Path
from rank_bm25 import BM25Okapi

from docling_core.types.doc.document import DoclingDocument, SectionHeaderItem, TitleItem, TextItem

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("constructor_contexto")

BM25_TOP_K = 5
CHUNK_WORDS = 250
OVERLAP_WORDS = 40

_META_RE = re.compile(
    r"(?i)(?:^\s*(?:edited|reviewed|correspondence)|^\s*(?:received|accepted|published)|"
    r"https?://doi\.org|\bdoi:\s*10\.|e-?mail:|@[a-z0-9.-]+\.[a-z]{2,}|\*correspondence|"
    r"\bcopyright\b|©\s*\d{4}|\bcreative\s+commons\b|\bopen-?access\s+article\b|"
    r"^\s*(?:issn|pmid|pmc)\b|\bfunding\s*:|\backnowledgments?\b|\bdata\s+availability\b|"
    r"^\s*reporting\s+summary\b|^\s*code\s+availability\b|^\s*author\s+contributions?\b|"
    r"^\s*competing\s+interests\b|^\s*additional\s+information\b)"
)

def _is_metadata(text: str) -> bool:
    t = text.strip()
    if not t: return True
    if len(t) > 500: return False # Salvavidas: si es un párrafo gigante, casi seguro es ciencia
    if _META_RE.search(t): return True
    return False

def _linearize_text_only(doc: DoclingDocument) -> list:
    """Extrae secuencialmente el texto puro del árbol Docling."""
    seq = []
    for item, _ in doc.iterate_items():
        if isinstance(item, SectionHeaderItem):
            seq.append({"type": "section", "text": item.text or ""})
        elif isinstance(item, TitleItem):
            seq.append({"type": "title", "text": item.text or ""})
        elif isinstance(item, TextItem):
            seq.append({"type": "text", "text": item.text or ""})
    return seq

def _extract_abstract(seq: list) -> str:
    # 1. Búsqueda estricta de la sección (evitando "Reporting Summary")
    for i, n in enumerate(seq):
        if n["type"] == "section" and re.search(r"^\s*(abstract|resumen)\b", n["text"], re.IGNORECASE):
            blocks = []
            for m in seq[i + 1:]:
                if m["type"] == "section": break
                if m["type"] == "text" and len(m["text"].strip()) >= 60 and not _is_metadata(m["text"]):
                    blocks.append(m["text"].strip())
                if len(blocks) >= 3: break
            if blocks: return "\n\n".join(blocks)
    
    # 2. Heurística de respaldo: El primer gran bloque de texto que hable de ciencia
    blocks = []
    started = False
    for n in seq:
        if n["type"] != "text": continue
        t = n["text"].strip()
        if _is_metadata(t): continue
        
        # En biología, un párrafo de introducción suele ser sustancioso (>250 chars)
        if not started and len(t) >= 250:
            started = True
            blocks.append(t)
        elif started:
            blocks.append(t)
            # Tomamos los primeros 2 o 3 párrafos como introducción/abstract
            if len(blocks) >= 2: break
            
    return "\n\n".join(blocks) if blocks else "Abstract no detectado."

def _chunk_paper(seq: list) -> list:
    paragraphs = [n["text"].strip() for n in seq if n["type"] == "text" and len(n["text"].strip()) > 80 and not _is_metadata(n["text"])]
    chunks = []
    current_words = []
    for para in paragraphs:
        words = para.split()
        if len(current_words) + len(words) > CHUNK_WORDS:
            if current_words: chunks.append(" ".join(current_words))
            current_words = current_words[-OVERLAP_WORDS:] + words
        else:
            current_words.extend(words)
    if current_words: chunks.append(" ".join(current_words))
    return chunks

def _tokenize(text: str) -> list:
    tokens = re.findall(r"[a-záéíóúñü0-9]+", text.lower())
    stops = {"the", "a", "an", "and", "or", "of", "in", "to", "for", "is", "are", "was", "were", "with", "that", "this", "from", "by", "on"}
    return [t for t in tokens if t not in stops and len(t) > 1]

def process_context(docling_json: Path, manifest_json: Path):
    stem = docling_json.stem.replace(".docling", "")
    out_dir = docling_json.parent
    
    with open(manifest_json, "r", encoding="utf-8") as f:
        visual_items = json.load(f)
        
    doc = DoclingDocument.load_from_json(docling_json)
    seq = _linearize_text_only(doc)
    
    abstract = _extract_abstract(seq)
    chunks = _chunk_paper(seq)
    
    tokenized_corpus = [_tokenize(c) for c in chunks]
    valid = [(i, tc) for i, tc in enumerate(tokenized_corpus) if tc]
    if valid:
        valid_indices, valid_tokens = zip(*valid)
        bm25 = BM25Okapi(list(valid_tokens))
    else:
        bm25 = None

    resultados = []
    for item in visual_items:
        caption = item.get("caption_oficial", "")
        
        # Query inteligente: Si Docling no encontró caption, busca menciones de la figura por su ID
        query = caption if caption else f"Figure {item['item_id'].replace('.png', '').split('item')[-1]}"
        
        bm25_passages = []
        if bm25:
            q_tokens = _tokenize(query)
            if q_tokens:
                scores = bm25.get_scores(q_tokens)
                ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
                for r in ranked[:BM25_TOP_K]:
                    if scores[r] > 0:
                        bm25_passages.append(chunks[valid_indices[r]])

        resultados.append({
            "item_id": item["item_id"],
            "crop_path": item["crop_path"],
            "caption_oficial": caption,
            "abstract_paper": abstract,
            "bm25_top_passages": bm25_passages
        })

    out_json = out_dir / "context.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)
        
    logger.info(f"✅ [{stem}] Contexto curado. {len(resultados)} imágenes. Abstract: {len(abstract)} chars. Chunks BM25: {len(chunks)}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_dir", type=str, required=True, help="Directorio del paper (contiene .docling.json y visual_manifest.json)")
    args = parser.parse_args()
    
    base_dir = Path(args.json_dir)
    docling_file = list(base_dir.glob("*.docling.json"))[0]
    manifest_file = base_dir / "visual_manifest.json"
    
    process_context(docling_file, manifest_file)

if __name__ == "__main__":
    main()