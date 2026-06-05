"""
extract_tables.py — Extractor de tablas científicas desde PDFs
===============================================================

Estrategia principal: PyMuPDF find_tables() (lines_strict → default fallback)
  - Filtro de área mínima (5000 pt²) para eliminar falsos positivos
  - Merge automático de tablas multi-página ("Continued")
  - Corrección de rotación vía PIL

Estrategia fallback: Camelot lattice (si PyMuPDF no detecta ninguna tabla)
  - Requiere: pip install camelot-py[base] + ghostscript

Por cada tabla produce:
  - PNG renderizado (corregido si está rotado)
  - Contenido estructurado (filas/columnas) en tables.json

Uso:
    python extract_tables.py paper.pdf --out extracted/
    python extract_tables.py paper.pdf --out extracted/ --dpi 250

Integración con pipeline:
    from extract_tables import extract_tables_all
    extract_tables_all("paper.pdf", out_dir="extracted/")
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import fitz  # PyMuPDF >= 1.23

# ─── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_DPI       = 200
DEFAULT_MIN_ROWS  = 2
DEFAULT_MIN_COLS  = 2
DEFAULT_MIN_CELLS = 6
MIN_AREA_PT2      = 5000   # filtro clave: elimina refs/autores mal detectados
CAPTION_SEARCH_PT = 80
ROTATION_RATIO    = 2.5


# ─── Caption cercano ───────────────────────────────────────────────────────────
CAPTION_RE = re.compile(
    r"^(Table|TABLE|Tabla|TABLA)\s+(\w+[\.\:]?)",
    re.IGNORECASE,
)

def find_nearby_caption(page, table_bbox, search_pt=CAPTION_SEARCH_PT) -> str:
    x0, y0, x1, y1 = table_bbox
    for zone in [
        (x0 - 20, y0 - search_pt, x1 + 20, y0),
        (x0 - 20, y1,             x1 + 20, y1 + search_pt),
    ]:
        text = page.get_text("text", clip=fitz.Rect(zone)).strip()
        for line in text.split("\n"):
            line = line.strip()
            if CAPTION_RE.match(line):
                return " ".join(l.strip() for l in text.split("\n") if l.strip())[:500]
    return ""


# ─── Rotación ──────────────────────────────────────────────────────────────────
def is_rotated(bbox) -> bool:
    x0, y0, x1, y1 = bbox
    w = x1 - x0
    h = y1 - y0
    return w > 0 and (h / w) > ROTATION_RATIO


# ─── Render ────────────────────────────────────────────────────────────────────
def render_table(page, bbox, out_path: Path, dpi: int, rotated: bool):
    mat  = fitz.Matrix(dpi / 72, dpi / 72)
    clip = fitz.Rect(bbox)
    pix  = page.get_pixmap(matrix=mat, clip=clip, alpha=False)
    if rotated:
        try:
            import PIL.Image, io
            img = PIL.Image.open(io.BytesIO(pix.tobytes("png")))
            img = img.rotate(90, expand=True)
            img.save(str(out_path))
            return (img.height, img.width)
        except ImportError:
            pass
    pix.save(str(out_path))
    return (pix.width, pix.height)


# ─── Markdown ──────────────────────────────────────────────────────────────────
def rows_to_markdown(rows: list[list[str | None]]) -> str:
    if not rows:
        return ""
    def clean(c):
        return str(c).replace("|", "\\|").replace("\n", " ").strip() if c else ""
    header = rows[0]
    lines  = ["| " + " | ".join(clean(c) for c in header) + " |",
              "|" + "|".join("---" for _ in header) + "|"]
    for row in rows[1:]:
        lines.append("| " + " | ".join(clean(c) for c in row) + " |")
    return "\n".join(lines)


# ─── Helpers ───────────────────────────────────────────────────────────────────
def _is_continued(rows) -> bool:
    """True si la primera fila indica que es continuación de la tabla anterior."""
    if not rows or not rows[0]:
        return False
    first = " ".join(str(c) for c in rows[0] if c).lower()
    return "continued" in first


def _passes_filters(bbox, rows, min_rows, min_cols, min_cells) -> bool:
    if not rows:
        return False
    nr    = len(rows)
    nc    = max(len(r) for r in rows) if rows else 0
    cells = sum(1 for row in rows for c in row if c and str(c).strip())
    area  = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    return (nr >= min_rows and nc >= min_cols
            and cells >= min_cells and area >= MIN_AREA_PT2)


# ─── Método principal: PyMuPDF find_tables() ──────────────────────────────────
def _extract_pymupdf(doc, min_rows, min_cols, min_cells) -> list[dict]:
    """
    Detecta tablas con lines_strict → fallback default.
    Aplica filtro de área y merge de páginas con 'Continued'.
    Devuelve lista de dicts con keys: pi, page_obj, bbox, rows_data, pages.
    """
    raw = []  # (pi, bbox, rows)
    for pi, page in enumerate(doc):
        try:
            found = page.find_tables(
                horizontal_strategy="lines_strict",
                vertical_strategy="lines_strict",
            ).tables
        except Exception:
            found = []
        if not found:
            try:
                found = page.find_tables().tables
            except Exception:
                found = []

        seen_bboxes = set()
        for tab in found:
            rows = tab.extract()
            bbox = tuple(round(v, 1) for v in tab.bbox)
            if bbox in seen_bboxes:
                continue
            seen_bboxes.add(bbox)
            if not _passes_filters(bbox, rows, min_rows, min_cols, min_cells):
                continue
            raw.append((pi, bbox, rows))

    # Merge: si primera fila dice "Continued" → unir con la tabla previa
    merged = []
    for pi, bbox, rows in raw:
        if _is_continued(rows) and merged:
            prev = merged[-1]
            prev["rows_data"].extend(rows[1:])  # saltar el header "Continued"
            prev["pages"].append(pi + 1)
        else:
            merged.append({
                "pi":        pi,
                "page_obj":  doc[pi],
                "bbox":      bbox,
                "rows_data": list(rows),
                "pages":     [pi + 1],
            })

    return merged


# ─── Método fallback: Camelot lattice ─────────────────────────────────────────
def _extract_camelot(pdf_path: str, doc, min_rows, min_cols, min_cells) -> list[dict]:
    """Fallback cuando PyMuPDF no detecta ninguna tabla con bordes."""
    try:
        import camelot
    except ImportError:
        return []

    results = []
    for pi, page in enumerate(doc):
        try:
            tbls = camelot.read_pdf(
                pdf_path, pages=str(pi + 1),
                flavor="lattice", suppress_stdout=True,
            )
        except Exception:
            continue
        for tbl in tbls:
            acc = tbl.parsing_report.get("accuracy", 0)
            if acc < 50:
                continue
            rows = [list(row) for row in tbl.df.values.tolist()]
            x1, y1, x2, y2 = tbl._bbox
            ph   = page.rect.height
            bbox = (round(x1, 1), round(ph - y2, 1),
                    round(x2, 1), round(ph - y1, 1))
            if not _passes_filters(bbox, rows, min_rows, min_cols, min_cells):
                continue
            results.append({
                "pi":        pi,
                "page_obj":  page,
                "bbox":      bbox,
                "rows_data": rows,
                "pages":     [pi + 1],
                "camelot_accuracy": acc,
            })
    return results


# ─── Pipeline principal ────────────────────────────────────────────────────────
def extract_tables_all(pdf_path: str, out_dir: str = "extracted",
                       dpi: int = DEFAULT_DPI,
                       min_rows: int = DEFAULT_MIN_ROWS,
                       min_cols: int = DEFAULT_MIN_COLS,
                       min_cells: int = DEFAULT_MIN_CELLS,
                       quiet: bool = False) -> list[dict]:

    pdf_path = Path(pdf_path)
    out_dir  = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(str(pdf_path))

    # 1) Intento principal: PyMuPDF fix
    candidates = _extract_pymupdf(doc, min_rows, min_cols, min_cells)

    # 2) Fallback: Camelot lattice si PyMuPDF no encontró nada
    if not candidates:
        if not quiet:
            print("  (PyMuPDF: 0 tablas → probando Camelot lattice...)")
        candidates = _extract_camelot(str(pdf_path), doc, min_rows, min_cols, min_cells)

    tables = []
    for m in candidates:
        pi        = m["pi"]
        page      = m["page_obj"]
        bbox      = m["bbox"]
        rows_data = m["rows_data"]
        pages     = m["pages"]

        rotated  = is_rotated(bbox)
        label    = f"Table_{len(tables) + 1}"
        filename = f"p{pi + 1:03d}_{label}.png"
        out_path = out_dir / filename

        img_size = render_table(page, bbox, out_path, dpi, rotated)

        caption = find_nearby_caption(page, bbox)

        clean_rows = [
            [str(c).strip() if c is not None else "" for c in row]
            for row in rows_data
        ]
        n_rows = len(clean_rows)
        n_cols = max(len(r) for r in clean_rows) if clean_rows else 0

        entry = {
            "label":      label,
            "kind":       "table",
            "page":       pi + 1,
            "pages":      pages,
            "bbox":       list(bbox),
            "caption":    caption,
            "image_path": str(out_path),
            "image_size": list(img_size),
            "rotated":    rotated,
            "row_count":  n_rows,
            "col_count":  n_cols,
            "rows":       clean_rows,
            "markdown":   rows_to_markdown(clean_rows),
        }
        if "camelot_accuracy" in m:
            entry["camelot_accuracy"] = m["camelot_accuracy"]

        tables.append(entry)

        if not quiet:
            method = "camelot" if "camelot_accuracy" in m else "pymupdf"
            rot_tag = " [rotada]" if rotated else ""
            pp_tag  = f" pp={'+'.join(str(p) for p in pages)}" if len(pages) > 1 else ""
            cap_tag = f" | {caption[:50]}" if caption else ""
            print(f"  p{pi+1} {label} [{method}]{rot_tag}{pp_tag}: "
                  f"{n_rows}x{n_cols}{cap_tag}")

    doc.close()

    out_json = out_dir / "tables.json"
    out_json.write_text(
        json.dumps({
            "pdf":   str(pdf_path),
            "total": len(tables),
            "items": tables,
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if not quiet:
        print(f"\n{len(tables)} tabla(s) -> {out_json}")

    return tables


# ─── CLI ───────────────────────────────────────────────────────────────────────
def main(argv=None):
    p = argparse.ArgumentParser(
        description="Extrae tablas de PDFs científicos (PyMuPDF fix + Camelot fallback)")
    p.add_argument("pdf",        help="PDF de entrada")
    p.add_argument("--out",      default="extracted")
    p.add_argument("--dpi",      type=int, default=DEFAULT_DPI)
    p.add_argument("--min-rows", type=int, default=DEFAULT_MIN_ROWS)
    p.add_argument("--min-cols", type=int, default=DEFAULT_MIN_COLS)
    p.add_argument("--min-cells",type=int, default=DEFAULT_MIN_CELLS)
    p.add_argument("--quiet",    action="store_true")
    args = p.parse_args(argv)

    if not Path(args.pdf).exists():
        sys.exit(f"Archivo no encontrado: {args.pdf}")

    print(f"Extrayendo tablas: {args.pdf}")
    tables = extract_tables_all(
        args.pdf,
        out_dir   = args.out,
        dpi       = args.dpi,
        min_rows  = args.min_rows,
        min_cols  = args.min_cols,
        min_cells = args.min_cells,
        quiet     = args.quiet,
    )
    print(f"Total: {len(tables)} tabla(s)")


if __name__ == "__main__":
    main()
