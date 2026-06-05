"""
Scientific Figure Extractor
============================

Extrae figuras y tablas individuales de PDFs científicos basándose en
la detección de captions ("Figure N", "Table N") y la geometría del
contenido visual en su columna.

A diferencia de renderizar páginas completas, este extractor produce
una imagen por figura/tabla, respetando layout multi-columna y
límites entre figuras consecutivas.

Uso:
    python extract_figures.py paper.pdf --out out/
    python extract_figures.py paper.pdf --out out/ --dpi 300

Salida:
    out/p00X_Figure_N.png    una imagen por figura/tabla
    out/figures.json          metadata: bbox, caption, página, ruta
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF


# ─── Configuración por defecto ───────────────────────────────────────────────
DEFAULT_DPI            = 200
DEFAULT_PADDING        = 16
DEFAULT_MAX_HEIGHT     = 9999  # sin límite práctico: captura figuras de página completa
CROSS_COLUMN_TOL       = 50    # tolerancia para clasificar columna izq/der/full
FULL_WIDTH_RATIO       = 0.45  # si visuals ocupan >45% del ancho → tratar como full-width

CAPTION_RE = re.compile(
    r'^\s*(Figure|Fig\.?|Table|Extended\s+Data\s+Fig(?:ure|\.)?)\s*(\d+)',
    re.IGNORECASE,
)


@dataclass
class Caption:
    page:  int      # 1-indexed
    label: str      # "Figure 3" / "Table 1" / "ExtFig 2"
    bbox:  tuple    # (x0, y0, x1, y1)
    text:  str      # texto completo del caption (recortado)
    kind:  str      # "figure" | "table"


# ─── Detección de captions ───────────────────────────────────────────────────
def find_captions(doc) -> list[Caption]:
    captions: list[Caption] = []
    for pi, page in enumerate(doc):
        for b in page.get_text("dict")["blocks"]:
            if b.get("type") != 0 or not b.get("lines"):
                continue
            first_line = "".join(s["text"] for s in b["lines"][0]["spans"]).strip()
            m = CAPTION_RE.match(first_line)
            if not m:
                continue
            # Ignorar bloques rotados: caption real nunca tiene <20pt de ancho
            bx0, _, bx1, _ = b["bbox"]
            if (bx1 - bx0) < 20:
                continue
            raw = m.group(1).strip().lower()
            if "table" in raw:
                kind, prefix = "table", "Table"
            elif "extended" in raw:
                kind, prefix = "figure", "ExtFig"
            else:
                kind, prefix = "figure", "Figure"
            label = f"{prefix} {m.group(2)}"
            full_text = " ".join(
                "".join(s["text"] for s in line["spans"])
                for line in b["lines"]
            )[:500]
            captions.append(Caption(
                page=pi + 1, label=label, bbox=tuple(b["bbox"]),
                text=full_text, kind=kind,
            ))
    return captions


# ─── Detección de columna ────────────────────────────────────────────────────
def get_column(bbox, page_width):
    x0, _, x1, _ = bbox
    mid = page_width / 2
    if x1 < mid + CROSS_COLUMN_TOL and x0 < mid:
        return "left"
    if x0 > mid - CROSS_COLUMN_TOL and x1 > mid:
        return "right"
    return "full"


def in_column(bbox, column, page_width):
    x0, _, x1, _ = bbox
    mid = page_width / 2
    if column == "left":
        return x1 <= mid + CROSS_COLUMN_TOL
    if column == "right":
        return x0 >= mid - CROSS_COLUMN_TOL
    return True


# ─── Bbox de figura ──────────────────────────────────────────────────────────
TEXT_LABEL_MARGIN = 60  # pt — margen para capturar etiquetas de ejes y paneles


def find_figure_region(page, caption, prev_boundary_y, max_height):
    pw = page.rect.width
    _, cy0, _, _ = caption.bbox
    column = get_column(caption.bbox, pw)

    visuals = []
    for b in page.get_text("dict")["blocks"]:
        if b.get("type") == 1:  # imagen embebida
            visuals.append(tuple(b["bbox"]))
    for d in page.get_drawings():
        r = d.get("rect")
        if r and r.width > 2 and r.height > 2:  # ignorar trazos/líneas muy finos
            visuals.append((r.x0, r.y0, r.x1, r.y1))

    # Candidatos sin filtro de columna (para detectar si es full-width)
    all_candidates = []
    for bb in visuals:
        bx0, by0, bx1, by1 = bb
        if by0 > cy0 + 5:              # excluir si empieza bajo el caption
            continue
        if by1 < prev_boundary_y - 5:  # excluir solo si está completamente sobre el boundary
            continue
        all_candidates.append(bb)

    if not all_candidates:
        return None

    # Detectar si el contenido es de ancho completo (multi-panel que abarca columnas)
    combined_x0 = min(b[0] for b in all_candidates)
    combined_x1 = max(b[2] for b in all_candidates)
    if (combined_x1 - combined_x0) > pw * FULL_WIDTH_RATIO:
        effective_column = "full"
    else:
        effective_column = column

    candidates = [bb for bb in all_candidates if in_column(bb, effective_column, pw)]
    if not candidates:
        candidates = all_candidates

    # Bbox visual inicial
    vx0 = min(b[0] for b in candidates)
    vy0 = min(b[1] for b in candidates)
    vx1 = max(b[2] for b in candidates)
    vy1 = max(b[3] for b in candidates)

    # Expandir con bloques de texto adyacentes: etiquetas de ejes, paneles (a, b, c…), ticks
    # Solo texto estrecho — etiquetas son cortas, párrafos de cuerpo no
    for b in page.get_text("dict")["blocks"]:
        if b.get("type") != 0:
            continue
        bb = tuple(b["bbox"])
        if tuple(b["bbox"]) == tuple(caption.bbox):
            continue
        bx0, by0, bx1, by1 = bb
        if by0 > cy0 + 5:
            continue
        if by1 < prev_boundary_y - 5:
            continue
        if (bx1 - bx0) > pw * 0.35:   # párrafo de cuerpo (ancho) → ignorar
            continue
        if (bx1 > vx0 - TEXT_LABEL_MARGIN and bx0 < vx1 + TEXT_LABEL_MARGIN and
                by1 > vy0 - TEXT_LABEL_MARGIN and by0 < vy1 + TEXT_LABEL_MARGIN):
            candidates.append(bb)

    return _combine_bboxes(candidates, page)


# ─── Bbox de tabla (es bloque de texto formateado) ──────────────────────────
def find_table_region(page, caption, prev_boundary_y, max_height):
    pw = page.rect.width
    _, cy0, _, _ = caption.bbox
    column = get_column(caption.bbox, pw)

    text_blocks = []
    for b in page.get_text("dict")["blocks"]:
        if b.get("type") != 0 or not b.get("lines"):
            continue
        bb = tuple(b["bbox"])
        if bb == caption.bbox:
            continue
        if bb[3] > cy0 + 5:
            continue
        if bb[1] < prev_boundary_y - 5:
            continue
        if cy0 - bb[3] > max_height:
            continue
        if not in_column(bb, column, pw):
            continue
        text_blocks.append(bb)

    if not text_blocks:
        return None

    text_blocks.sort(key=lambda b: -b[3])  # ordenar por cercanía al caption
    gathered = [text_blocks[0]]
    for b in text_blocks[1:]:
        if gathered[-1][1] - b[3] < 30:    # gap pequeño → mismo bloque
            gathered.append(b)
        else:
            break
    return _combine_bboxes(gathered, page)


def _fallback_region(page, caption, prev_boundary_y, white_threshold=0.97):
    """Renderiza el área sobre el caption y la retorna si tiene contenido no-blanco."""
    pw, ph = page.rect.width, page.rect.height
    cy0 = caption.bbox[1]
    col = get_column(caption.bbox, pw)
    if col == "left":
        x0, x1 = 0.0, pw / 2
    elif col == "right":
        x0, x1 = pw / 2, pw
    else:
        x0, x1 = 0.0, pw

    clip = fitz.Rect(x0, max(0, prev_boundary_y), x1, cy0)
    if clip.height < 10:
        return None

    mat = fitz.Matrix(72 / 72, 72 / 72)  # baja resolución para la comprobación
    pix = page.get_pixmap(matrix=mat, clip=clip, alpha=False)
    samples = pix.samples
    n = pix.width * pix.height
    if n == 0:
        return None
    white = sum(
        1 for i in range(0, len(samples), 3)
        if samples[i] > 240 and samples[i + 1] > 240 and samples[i + 2] > 240
    )
    if white / n > white_threshold:
        return None  # región esencialmente en blanco
    return (
        max(0, x0 - DEFAULT_PADDING),
        max(0, clip.y0 - DEFAULT_PADDING),
        min(pw, x1 + DEFAULT_PADDING),
        min(ph, cy0 + DEFAULT_PADDING),
    )


def _combine_bboxes(boxes, page):
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[2] for b in boxes)
    y1 = max(b[3] for b in boxes)
    pw, ph = page.rect.width, page.rect.height
    return (
        max(0, x0 - DEFAULT_PADDING),
        max(0, y0 - DEFAULT_PADDING),
        min(pw, x1 + DEFAULT_PADDING),
        min(ph, y1 + DEFAULT_PADDING),
    )


# ─── Render ──────────────────────────────────────────────────────────────────
def render_region(page, bbox, out_path, dpi):
    clip = fitz.Rect(*bbox)
    mat  = fitz.Matrix(dpi / 72, dpi / 72)
    pix  = page.get_pixmap(matrix=mat, clip=clip, alpha=False)
    pix.save(str(out_path))
    return clip.width, clip.height


# ─── Pipeline ────────────────────────────────────────────────────────────────
def extract_all(pdf_path, out_dir, dpi=DEFAULT_DPI, max_height=DEFAULT_MAX_HEIGHT, quiet=False):
    pdf_path = Path(pdf_path)
    out_dir  = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def log(msg):
        if not quiet:
            print(msg)

    doc = fitz.open(str(pdf_path))
    captions = find_captions(doc)
    log(f"Captions detectados: {len(captions)}")

    by_page = {}
    for c in captions:
        by_page.setdefault(c.page, []).append(c)

    results = []
    for cap in captions:
        page = doc[cap.page - 1]

        # Boundary previo: caption anterior en la MISMA columna de la página
        prev_y = 0.0
        col = get_column(cap.bbox, page.rect.width)
        for other in by_page[cap.page]:
            if other is cap:
                continue
            if get_column(other.bbox, page.rect.width) != col:
                continue
            if other.bbox[3] < cap.bbox[1] and other.bbox[3] > prev_y:
                prev_y = other.bbox[3]

        render_page = page  # página desde la que se renderizará la figura
        if cap.kind == "figure":
            region = find_figure_region(page, cap, prev_y, max_height)
        else:
            region = find_table_region(page, cap, prev_y, max_height)

        # Fallback 1: figura en la página SIGUIENTE (caption antes que la figura)
        if region is None and cap.page < len(doc):
            next_page = doc[cap.page]  # cap.page es 1-indexed → doc[cap.page] es la siguiente
            next_caps = by_page.get(cap.page + 1, [])
            boundary_y = min(c.bbox[1] for c in next_caps) if next_caps else next_page.rect.height
            syn_cap = Caption(
                page=cap.page + 1, label=cap.label,
                bbox=(0, boundary_y, next_page.rect.width, boundary_y),
                text=cap.text, kind=cap.kind,
            )
            region = find_figure_region(next_page, syn_cap, 0, max_height)
            if region is not None:
                render_page = next_page
                log(f"  [{cap.label}] p{cap.page} -> figura en página siguiente")

        # Fallback 2: Form XObject u otro contenido que PyMuPDF no enumera pero sí renderiza
        if region is None:
            region = _fallback_region(page, cap, prev_y)
            if region is not None:
                log(f"  [{cap.label}] p{cap.page} -> fallback render (Form XObject?)")

        if region is None:
            log(f"  [{cap.label}] p{cap.page} - sin región visual, skip")
            continue

        safe = cap.label.replace(" ", "_").replace(".", "")
        fname = f"p{cap.page:03d}_{safe}.png"
        out_path = out_dir / fname
        w, h = render_region(render_page, region, out_path, dpi)
        size_kb = out_path.stat().st_size / 1024
        log(f"  [{cap.label:14}] p{cap.page} -> {fname} ({w:.0f}x{h:.0f}px, {size_kb:.1f}KB)")

        results.append({
            "label":      cap.label,
            "kind":       cap.kind,
            "page":       cap.page,
            "bbox":       [round(v, 1) for v in region],
            "caption":    cap.text,
            "image_path": str(out_path),
            "image_size": [round(w), round(h)],
        })

    doc.close()

    meta_path = out_dir / "figures.json"
    meta_path.write_text(
        json.dumps({"pdf": str(pdf_path), "total": len(results), "items": results},
                   indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log(f"\nMetadata: {meta_path}")
    log(f"Total extraído: {len(results)} de {len(captions)} captions")
    return results


# ─── CLI ─────────────────────────────────────────────────────────────────────
def main(argv=None):
    p = argparse.ArgumentParser(
        description="Extrae figuras y tablas individuales de PDFs científicos.",
    )
    p.add_argument("pdf",  help="ruta al PDF de entrada")
    p.add_argument("--out", default="extracted", help="directorio de salida")
    p.add_argument("--dpi", type=int, default=DEFAULT_DPI, help="resolución del render (def: 200)")
    p.add_argument("--max-height", type=int, default=DEFAULT_MAX_HEIGHT,
                   help="altura máxima del bbox de una figura en pt (def: 600)")
    p.add_argument("--quiet", action="store_true", help="sin output a stdout")
    args = p.parse_args(argv)

    if not Path(args.pdf).exists():
        sys.exit(f"PDF no encontrado: {args.pdf}")

    extract_all(args.pdf, args.out, dpi=args.dpi, max_height=args.max_height, quiet=args.quiet)


if __name__ == "__main__":
    main()
