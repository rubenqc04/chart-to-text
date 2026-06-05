import os
import re
import json
import logging
import argparse
from pathlib import Path
from typing import Any
from rank_bm25 import BM25Okapi

from docling_core.types.doc.document import DoclingDocument, SectionHeaderItem, TitleItem, TextItem
from docling_core.types.doc.base import CoordOrigin

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("constructor_contexto_v2")

BM25_TOP_K = 8
CHUNK_WORDS = 250
OVERLAP_WORDS = 40
NEIGHBOR_PARAGRAPHS = 3
SPATIAL_MARGIN_PT = 140.0
MAX_CONTEXT_PASSAGES_FOR_PROMPT = 8
MAX_PASSAGE_CHARS = 1200

_META_RE = re.compile(
    r"(?i)(?:^\s*(?:edited|reviewed|correspondence)|^\s*(?:received|accepted|published)|"
    r"https?://doi\.org|\bdoi:\s*10\.|e-?mail:|@[a-z0-9.-]+\.[a-z]{2,}|\*correspondence|"
    r"\bcopyright\b|©\s*\d{4}|\bcreative\s+commons\b|\bopen-?access\s+article\b|"
    r"^\s*(?:issn|pmid|pmc)\b|\bfunding\s*:|\backnowledgments?\b|\bdata\s+availability\b|"
    r"^\s*reporting\s+summary\b|^\s*code\s+availability\b|^\s*author\s+contributions?\b|"
    r"^\s*competing\s+interests\b|^\s*additional\s+information\b|\breferences\b$)"
)

FIG_LABEL_RE = re.compile(
    r"(?i)\b("
    r"(?:fig(?:ure)?\.?\s*\d+[a-z]?)"
    r"|(?:extended\s+data\s+fig(?:ure)?\.?\s*\d+[a-z]?)"
    r"|(?:supplementary\s+fig(?:ure)?\.?\s*s?\d+[a-z]?)"
    r"|(?:table\s*s?\d+[a-z]?)"
    r"|(?:supplementary\s+table\s*s?\d+[a-z]?)"
    r")\b"
)

_SECTION_PRIORITY_PATTERNS = [
    (re.compile(r"(?i)\b(results?|findings?|experiments?|evaluation|analysis)\b"), 1.0),
    (re.compile(r"(?i)\b(methods?|materials?|experimental|protocol)\b"), 0.7),
    (re.compile(r"(?i)\b(introduction|background)\b"), 0.35),
    (re.compile(r"(?i)\b(discussion|conclusion)\b"), 0.55),
]


def _is_metadata(text: str) -> bool:
    t = text.strip()
    if not t:
        return True
    if len(t) > 500:
        return False
    return bool(_META_RE.search(t))


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _truncate(text: str, limit: int = MAX_PASSAGE_CHARS) -> str:
    text = _clean_text(text)
    return text if len(text) <= limit else text[:limit] + " [TRUNCATED]"


def _bbox_from_prov(prov, page_height: float) -> list[float] | None:
    if not prov or not getattr(prov, "bbox", None):
        return None
    bb = prov.bbox
    if bb.coord_origin == CoordOrigin.BOTTOMLEFT:
        bb = bb.to_top_left_origin(page_height=page_height)
    return [float(bb.l), float(bb.t), float(bb.r), float(bb.b)]


def _linearize_with_layout(doc: DoclingDocument) -> list[dict[str, Any]]:
    """Linealiza texto manteniendo orden, sección, página y bbox.

    Esto reemplaza a _linearize_text_only porque el contexto por figura necesita
    ubicar párrafos alrededor de una caption/figura y no solo buscar por BM25.
    """
    seq = []
    current_section = None

    for idx, (item, _) in enumerate(doc.iterate_items()):
        text = getattr(item, "text", None)
        if not text:
            continue
        text = _clean_text(text)
        if not text:
            continue
        page = None
        bbox = None
        if getattr(item, "prov", None):
            try:
                prov = item.prov[0]
                page = prov.page_no
                page_h = doc.pages[prov.page_no].size.height
                bbox = _bbox_from_prov(prov, page_h)
            except Exception:
                pass

        if isinstance(item, SectionHeaderItem):
            current_section = text
            seq.append({
                "idx": idx,
                "type": "section",
                "text": text,
                "page": page,
                "bbox": bbox,
                "section": current_section,
            })
        elif isinstance(item, TitleItem):
            seq.append({
                "idx": idx,
                "type": "title",
                "text": text,
                "page": page,
                "bbox": bbox,
                "section": current_section,
            })
        elif isinstance(item, TextItem):
            seq.append({
                "idx": idx,
                "type": "text",
                "text": text,
                "page": page,
                "bbox": bbox,
                "section": current_section,
            })

    return seq


def _extract_abstract(seq: list[dict[str, Any]]) -> str:
    for i, n in enumerate(seq):
        if n["type"] == "section" and re.search(r"^\s*(abstract|resumen)\b", n["text"], re.IGNORECASE):
            blocks = []
            for m in seq[i + 1:]:
                if m["type"] == "section":
                    break
                if m["type"] == "text" and len(m["text"].strip()) >= 60 and not _is_metadata(m["text"]):
                    blocks.append(m["text"].strip())
                if len(blocks) >= 3:
                    break
            if blocks:
                return "\n\n".join(blocks)

    blocks = []
    started = False
    for n in seq:
        if n["type"] != "text":
            continue
        t = n["text"].strip()
        if _is_metadata(t):
            continue
        if not started and len(t) >= 250:
            started = True
            blocks.append(t)
        elif started:
            blocks.append(t)
            if len(blocks) >= 2:
                break
    return "\n\n".join(blocks) if blocks else "Abstract no detectado."


def _tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-záéíóúñü0-9]+", text.lower())
    stops = {
        "the", "a", "an", "and", "or", "of", "in", "to", "for", "is", "are", "was", "were",
        "with", "that", "this", "from", "by", "on", "as", "at", "be", "it", "we", "our", "their",
        "la", "el", "los", "las", "de", "del", "y", "o", "en", "para", "con", "por", "un", "una",
    }
    return [t for t in tokens if t not in stops and len(t) > 1]


def _section_priority(section: str | None) -> float:
    if not section:
        return 0.0
    for pattern, score in _SECTION_PRIORITY_PATTERNS:
        if pattern.search(section):
            return score
    return 0.2


def _chunk_paper(seq: list[dict[str, Any]]) -> list[dict[str, Any]]:
    paragraphs = [
        n for n in seq
        if n["type"] == "text" and len(n["text"].strip()) > 80 and not _is_metadata(n["text"])
    ]
    chunks = []
    current_words = []
    current_nodes = []

    def flush():
        if not current_words:
            return
        text = " ".join(current_words)
        pages = sorted({n["page"] for n in current_nodes if n.get("page") is not None})
        sections = []
        for n in current_nodes:
            sec = n.get("section")
            if sec and sec not in sections:
                sections.append(sec)
        chunks.append({
            "chunk_id": len(chunks),
            "text": text,
            "start_idx": current_nodes[0]["idx"] if current_nodes else None,
            "end_idx": current_nodes[-1]["idx"] if current_nodes else None,
            "pages": pages,
            "sections": sections,
            "section_priority": max([_section_priority(s) for s in sections], default=0.0),
        })

    for node in paragraphs:
        words = node["text"].split()
        if len(current_words) + len(words) > CHUNK_WORDS:
            flush()
            overlap = current_words[-OVERLAP_WORDS:] if current_words else []
            current_words = overlap + words
            current_nodes = current_nodes[-1:] + [node] if current_nodes else [node]
        else:
            current_words.extend(words)
            current_nodes.append(node)

    flush()
    return chunks


def _extract_figure_label(caption: str, visible_text: str = "") -> str | None:
    text = f"{caption}\n{visible_text}"
    m = FIG_LABEL_RE.search(text or "")
    if not m:
        return None
    return _clean_text(m.group(1))


def _figure_reference_regex(label: str | None) -> re.Pattern | None:
    if not label:
        return None
    label_clean = _clean_text(label)
    m = re.search(r"(?i)(\d+[a-z]?)", label_clean)
    if not m:
        return re.compile(re.escape(label_clean), re.IGNORECASE)
    number = re.escape(m.group(1))

    if re.search(r"(?i)table", label_clean):
        return re.compile(rf"(?i)\b(?:supplementary\s+)?table\s*{number}\b")
    if re.search(r"(?i)extended\s+data", label_clean):
        return re.compile(rf"(?i)\bextended\s+data\s+fig(?:ure)?\.?\s*{number}\b")
    if re.search(r"(?i)supplementary", label_clean):
        return re.compile(rf"(?i)\bsupplementary\s+fig(?:ure)?\.?\s*s?{number}\b")
    return re.compile(rf"(?i)\bfig(?:ure)?\.?\s*{number}\b")


def _find_caption_anchor_idx(seq: list[dict[str, Any]], item: dict[str, Any]) -> int | None:
    caption = _clean_text(item.get("caption_oficial", ""))
    if caption:
        caption_short = caption[:180]
        for n in seq:
            if n["type"] in {"text", "title"} and caption_short and caption_short in n["text"]:
                return n["idx"]

    page = item.get("page")
    fig_bbox = item.get("bbox_expanded") or item.get("bbox_original") or item.get("bbox_rendered")
    candidates = [n for n in seq if n["type"] == "text" and n.get("page") == page and n.get("bbox")]
    if not candidates:
        return None
    if not fig_bbox:
        return candidates[0]["idx"]

    fx0, fy0, fx1, fy1 = fig_bbox
    fcx, fcy = (fx0 + fx1) / 2.0, (fy0 + fy1) / 2.0

    def dist(n):
        x0, y0, x1, y1 = n["bbox"]
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        return abs(cx - fcx) * 0.25 + abs(cy - fcy)

    return min(candidates, key=dist)["idx"]


def _neighbor_paragraphs(seq: list[dict[str, Any]], anchor_idx: int | None, before: int = NEIGHBOR_PARAGRAPHS, after: int = NEIGHBOR_PARAGRAPHS):
    texts = [n for n in seq if n["type"] == "text" and len(n["text"]) > 60 and not _is_metadata(n["text"])]
    if anchor_idx is None or not texts:
        return {"before": [], "after": []}

    pos = min(range(len(texts)), key=lambda i: abs(texts[i]["idx"] - anchor_idx))
    return {
        "before": texts[max(0, pos - before):pos],
        "after": texts[pos + 1:pos + 1 + after],
    }


def _rect_distance(a: list[float], b: list[float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    acx, acy = (ax0 + ax1) / 2.0, (ay0 + ay1) / 2.0
    bcx, bcy = (bx0 + bx1) / 2.0, (by0 + by1) / 2.0
    return abs(acx - bcx) * 0.25 + abs(acy - bcy)


def _same_page_spatial_context(seq: list[dict[str, Any]], item: dict[str, Any], limit: int = 5) -> list[dict[str, Any]]:
    page = item.get("page")
    fig_bbox = item.get("bbox_expanded") or item.get("bbox_original") or item.get("bbox_rendered")
    if not page or not fig_bbox:
        return []

    x0, y0, x1, y1 = fig_bbox
    expanded = [x0 - SPATIAL_MARGIN_PT, y0 - SPATIAL_MARGIN_PT, x1 + SPATIAL_MARGIN_PT, y1 + SPATIAL_MARGIN_PT]

    def intersects(a, b):
        return not (a[2] < b[0] or a[0] > b[2] or a[3] < b[1] or a[1] > b[3])

    candidates = []
    for n in seq:
        if n["type"] != "text" or n.get("page") != page or not n.get("bbox"):
            continue
        if len(n["text"]) < 50 or _is_metadata(n["text"]):
            continue
        if intersects(n["bbox"], expanded):
            n2 = dict(n)
            n2["distance_to_figure"] = _rect_distance(n["bbox"], fig_bbox)
            candidates.append(n2)

    candidates.sort(key=lambda n: n["distance_to_figure"])
    return candidates[:limit]


def _find_reference_mentions(seq: list[dict[str, Any]], figure_label: str | None, limit: int = 8) -> list[dict[str, Any]]:
    pattern = _figure_reference_regex(figure_label)
    if not pattern:
        return []
    hits = []
    for n in seq:
        if n["type"] != "text" or len(n["text"]) < 40 or _is_metadata(n["text"]):
            continue
        if pattern.search(n["text"]):
            hits.append(n)
        if len(hits) >= limit:
            break
    return hits


def _bm25_search(chunks: list[dict[str, Any]], query: str, top_k: int = BM25_TOP_K) -> list[dict[str, Any]]:
    if not chunks or not _tokenize(query):
        return []

    tokenized_corpus = [_tokenize(c["text"]) for c in chunks]
    valid = [(i, tc) for i, tc in enumerate(tokenized_corpus) if tc]
    if not valid:
        return []

    valid_indices, valid_tokens = zip(*valid)
    bm25 = BM25Okapi(list(valid_tokens))
    scores = bm25.get_scores(_tokenize(query))

    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    hits = []
    for r in ranked:
        if len(hits) >= top_k:
            break
        if scores[r] <= 0:
            continue
        chunk = chunks[valid_indices[r]]
        # Pequeño boost informativo por sección; no cambia BM25, solo lo documenta.
        hybrid_score = float(scores[r]) + 0.2 * float(chunk.get("section_priority", 0.0))
        hits.append({
            "chunk_id": chunk["chunk_id"],
            "text": chunk["text"],
            "score_bm25": float(scores[r]),
            "score_hybrid_hint": hybrid_score,
            "pages": chunk.get("pages", []),
            "sections": chunk.get("sections", []),
        })
    return hits


def _context_entry(source: str, node_or_text: Any, score: float = 0.0) -> dict[str, Any]:
    if isinstance(node_or_text, dict):
        return {
            "source": source,
            "text": _truncate(node_or_text.get("text", "")),
            "idx": node_or_text.get("idx"),
            "page": node_or_text.get("page"),
            "section": node_or_text.get("section"),
            "score": score,
        }
    return {"source": source, "text": _truncate(str(node_or_text)), "score": score}


def _dedupe_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    out = []
    for e in entries:
        text = _clean_text(e.get("text", ""))
        if not text:
            continue
        key = text[:220].lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(e)
    return out


def _build_hybrid_context(seq: list[dict[str, Any]], chunks: list[dict[str, Any]], item: dict[str, Any]) -> dict[str, Any]:
    caption = _clean_text(item.get("caption_oficial", ""))
    visible_text = _clean_text(item.get("visible_text_from_crop", ""))
    figure_label = item.get("figure_label_guess") or _extract_figure_label(caption, visible_text)
    anchor_idx = _find_caption_anchor_idx(seq, item)

    neighbors = _neighbor_paragraphs(seq, anchor_idx)
    spatial = _same_page_spatial_context(seq, item)
    mentions = _find_reference_mentions(seq, figure_label)

    bm25_query_parts = [caption, figure_label or "", visible_text[:800]]
    bm25_query = "\n".join(p for p in bm25_query_parts if p)
    bm25_hits = _bm25_search(chunks, bm25_query, top_k=BM25_TOP_K)

    prioritized = []
    if caption:
        prioritized.append(_context_entry("official_caption", caption, score=3.0))
    for n in mentions:
        prioritized.append(_context_entry("exact_figure_reference", n, score=2.5 + _section_priority(n.get("section"))))
    for n in neighbors["before"]:
        prioritized.append(_context_entry("nearby_before", n, score=1.8 + _section_priority(n.get("section"))))
    for n in neighbors["after"]:
        prioritized.append(_context_entry("nearby_after", n, score=1.6 + _section_priority(n.get("section"))))
    for n in spatial:
        prioritized.append(_context_entry("same_page_spatial", n, score=1.4 + _section_priority(n.get("section"))))
    for h in bm25_hits:
        prioritized.append({
            "source": "bm25_caption_visible_text",
            "text": _truncate(h["text"]),
            "score": h["score_hybrid_hint"],
            "score_bm25": h["score_bm25"],
            "chunk_id": h["chunk_id"],
            "pages": h.get("pages", []),
            "sections": h.get("sections", []),
        })

    prioritized = _dedupe_entries(prioritized)
    # Mantener primero las fuentes más confiables; BM25 puede tener score alto pero no debe superar caption/menciones.
    source_order = {
        "official_caption": 0,
        "exact_figure_reference": 1,
        "nearby_before": 2,
        "nearby_after": 3,
        "same_page_spatial": 4,
        "bm25_caption_visible_text": 5,
    }
    prioritized.sort(key=lambda e: (source_order.get(e["source"], 99), -float(e.get("score", 0))))

    return {
        "figure_label": figure_label,
        "anchor_idx": anchor_idx,
        "caption": caption,
        "visible_text_from_crop": visible_text,
        "nearby_before": [_context_entry("nearby_before", n) for n in neighbors["before"]],
        "nearby_after": [_context_entry("nearby_after", n) for n in neighbors["after"]],
        "same_page_spatial": [_context_entry("same_page_spatial", n) for n in spatial],
        "exact_figure_reference_mentions": [_context_entry("exact_figure_reference", n) for n in mentions],
        "bm25_caption_hits": [
            {
                "text": _truncate(h["text"]),
                "score_bm25": h["score_bm25"],
                "chunk_id": h["chunk_id"],
                "pages": h.get("pages", []),
                "sections": h.get("sections", []),
            }
            for h in bm25_hits
        ],
        "prioritized_context": prioritized[:MAX_CONTEXT_PASSAGES_FOR_PROMPT],
    }


def process_context(docling_json: Path, manifest_json: Path):
    stem = docling_json.stem.replace(".docling", "")
    out_dir = docling_json.parent

    with open(manifest_json, "r", encoding="utf-8") as f:
        visual_items = json.load(f)

    doc = DoclingDocument.load_from_json(docling_json)
    seq = _linearize_with_layout(doc)
    abstract = _extract_abstract(seq)
    chunks = _chunk_paper(seq)

    resultados = []
    for item in visual_items:
        hybrid = _build_hybrid_context(seq, chunks, item)

        # Compatibilidad con tus scripts 03 actuales: siguen leyendo bm25_top_passages.
        # En v2 esta lista contiene el contexto híbrido priorizado, no solo BM25 puro.
        hybrid_passages = [e["text"] for e in hybrid["prioritized_context"]]

        resultados.append({
            "item_id": item["item_id"],
            "crop_path": item["crop_path"],
            "page": item.get("page"),
            "caption_oficial": item.get("caption_oficial", ""),
            "figure_label": hybrid.get("figure_label"),
            "abstract_paper": abstract,
            "visible_text_from_crop": item.get("visible_text_from_crop", ""),
            "bbox_original": item.get("bbox_original"),
            "bbox_expanded": item.get("bbox_expanded"),
            "bbox_rendered": item.get("bbox_rendered"),
            "context_by_source": hybrid,
            "bm25_top_passages": hybrid_passages,
            "context_retrieval_metadata": {
                "version": "context_v2_hybrid_pdf_only",
                "strategies": [
                    "official_caption",
                    "nearby_before_after",
                    "same_page_spatial",
                    "exact_figure_reference_regex",
                    "bm25_caption_visible_text",
                ],
                "num_chunks": len(chunks),
                "num_prioritized_context_passages": len(hybrid_passages),
                "has_exact_figure_reference": len(hybrid.get("exact_figure_reference_mentions", [])) > 0,
                "has_visible_text_from_crop": bool(item.get("visible_text_from_crop")),
            },
        })

    out_json = out_dir / "context.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)

    logger.info(
        f"✅ [{stem}] Contexto híbrido v2. {len(resultados)} imágenes. "
        f"Abstract: {len(abstract)} chars. Chunks: {len(chunks)} → {out_json}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_dir", type=str, required=True, help="Directorio del paper con .docling.json y visual_manifest.json")
    args = parser.parse_args()

    base_dir = Path(args.json_dir)
    docling_candidates = list(base_dir.glob("*.docling.json"))
    if not docling_candidates:
        raise FileNotFoundError(f"No encontré .docling.json en {base_dir}")
    docling_file = docling_candidates[0]
    manifest_file = base_dir / "visual_manifest.json"
    if not manifest_file.exists():
        raise FileNotFoundError(f"No encontré visual_manifest.json en {base_dir}")

    process_context(docling_file, manifest_file)


if __name__ == "__main__":
    main()
