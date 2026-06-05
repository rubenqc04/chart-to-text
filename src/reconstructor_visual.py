import os
import json
import logging
import argparse
from pathlib import Path
import fitz  # PyMuPDF

from docling_core.types.doc.document import DoclingDocument, PictureItem
from docling_core.types.doc.base import BoundingBox, CoordOrigin

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("reconstructor_visual")

# --- Parámetros de Geometría y Limpieza ---
RENDER_DPI = 300
CROP_PADDING_PT = 5.0
TEXT_LABEL_MARGIN = 50.0  
CLUSTER_GAP_FRAC = 0.06
MIN_PERP_OVERLAP = 0.15
MIN_AREA_FRAC = 0.015       # Filtra logos pequeños (<1.5% de la página)
MAX_ASPECT_RATIO = 12.0     # Filtra líneas/banners decorativos
HEADER_FOOTER_BAND = 0.08   # Filtra logos en el margen superior/inferior

def _bbox_topleft(prov, page_height: float) -> BoundingBox:
    bb = prov.bbox
    if bb.coord_origin == CoordOrigin.BOTTOMLEFT:
        bb = bb.to_top_left_origin(page_height=page_height)
    return bb

def _vertical_gap(a: BoundingBox, b: BoundingBox) -> float:
    if a.b <= b.t: return b.t - a.b
    if b.b <= a.t: return a.t - b.b
    return 0.0

def _horizontal_gap(a: BoundingBox, b: BoundingBox) -> float:
    if a.r <= b.l: return b.l - a.r
    if b.r <= a.l: return a.l - b.r
    return 0.0

def _perp_overlap_frac(a: BoundingBox, b: BoundingBox, axis: str) -> float:
    if axis == "vertical":
        lo, hi = max(a.l, b.l), min(a.r, b.r)
        span = min(a.r - a.l, b.r - b.l)
    else:
        lo, hi = max(a.t, b.t), min(a.b, b.b)
        span = min(a.b - a.t, b.b - b.t)
    if span <= 0: return 0.0
    return max(0.0, hi - lo) / span

def _are_contiguous(a: BoundingBox, b: BoundingBox, page_w: float, page_h: float) -> bool:
    """Evalúa si dos fragmentos pertenecen a la misma figura."""
    vgap = _vertical_gap(a, b)
    hgap = _horizontal_gap(a, b)
    if vgap <= CLUSTER_GAP_FRAC * page_h and _perp_overlap_frac(a, b, "vertical") >= MIN_PERP_OVERLAP:
        return True
    if hgap <= CLUSTER_GAP_FRAC * page_w and _perp_overlap_frac(a, b, "horizontal") >= MIN_PERP_OVERLAP:
        return True
    return False

def _is_garbage(bbox: BoundingBox, page_w: float, page_h: float) -> tuple[bool, str]:
    """Identifica basura visual: logos, íconos o separadores."""
    w, h = bbox.r - bbox.l, bbox.b - bbox.t
    if w <= 0 or h <= 0:
        return True, "bbox_vacia"
    area_frac = (w * h) / (page_w * page_h)
    if area_frac < MIN_AREA_FRAC:
        return True, f"area_muy_pequena ({area_frac:.4f})"
    ar = max(w / h, h / w)
    if ar > MAX_ASPECT_RATIO:
        return True, f"aspect_ratio_extremo ({ar:.1f})"
    in_header = bbox.t < HEADER_FOOTER_BAND * page_h
    in_footer = bbox.b > (1 - HEADER_FOOTER_BAND) * page_h
    if (in_header or in_footer) and area_frac < 0.05:
        return True, "logo_cabecera_pie"
    return False, ""

def _expandir_bbox_con_textos(bbox: list, page: fitz.Page) -> list:
    """Expande el área para atrapar leyendas de ejes y números."""
    x0, y0, x1, y1 = bbox
    pw, ph = page.rect.width, page.rect.height
    
    search_rect = fitz.Rect(
        max(0, x0 - TEXT_LABEL_MARGIN), max(0, y0 - TEXT_LABEL_MARGIN),
        min(pw, x1 + TEXT_LABEL_MARGIN), min(ph, y1 + TEXT_LABEL_MARGIN)
    )
    
    nx0, ny0, nx1, ny1 = x0, y0, x1, y1
    
    for b in page.get_text("dict")["blocks"]:
        if b.get("type") != 0: continue  # Solo texto
        bx0, by0, bx1, by1 = b["bbox"]
        if (bx1 - bx0) > pw * 0.35: continue  # Ignorar párrafos de texto principal
        
        if fitz.Rect(bx0, by0, bx1, by1).intersects(search_rect):
            nx0, ny0, nx1, ny1 = min(nx0, bx0), min(ny0, by0), max(nx1, bx1), max(ny1, by1)
            
    return [nx0, ny0, nx1, ny1]

def reconstruir_figuras_y_markdown(pdf_path: Path, docling_json: Path, md_path: Path):
    stem = pdf_path.stem
    out_crops_dir = docling_json.parent / "crops"
    out_crops_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"[{stem}] Cargando documentos...")
    doc = DoclingDocument.load_from_json(docling_json)
    pdf_doc = fitz.open(str(pdf_path))
    
    # 1. Extraer fragmentos crudos
    raws = []
    for item, _ in doc.iterate_items():
        if isinstance(item, PictureItem) and item.prov:
            prov = item.prov[0]
            page_h = doc.pages[prov.page_no].size.height
            raws.append({
                "page": prov.page_no,
                "bbox": _bbox_topleft(prov, page_h)
            })
            
    # 2. Agrupar fragmentos por proximidad espacial (Union-Find)
    by_page = {}
    for r in raws:
        by_page.setdefault(r["page"], []).append(r)
        
    clusters = []
    for page_no, pics in by_page.items():
        page_w, page_h = doc.pages[page_no].size.width, doc.pages[page_no].size.height
        parent = list(range(len(pics)))
        
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
            
        def union(x, y):
            parent[find(x)] = find(y)
            
        for i in range(len(pics)):
            for j in range(i + 1, len(pics)):
                if _are_contiguous(pics[i]["bbox"], pics[j]["bbox"], page_w, page_h):
                    union(i, j)
                    
        groups = {}
        for idx, p in enumerate(pics):
            groups.setdefault(find(idx), []).append(p)
            
        for group in groups.values():
            # Bbox envolvente de todo el grupo
            bboxes = [p["bbox"] for p in group]
            merged_bbox = BoundingBox.enclosing_bbox(bboxes)
            clusters.append({
                "page": page_no,
                "bbox": merged_bbox,
                "n_frags": len(group)
            })

    # 3. Filtrar basura y recortar
    figuras_validas = []
    fig_idx = 0
    
    for cluster in clusters:
        page_no = cluster["page"]
        page_w, page_h = doc.pages[page_no].size.width, doc.pages[page_no].size.height
        
        es_basura, razon = _is_garbage(cluster["bbox"], page_w, page_h)
        if es_basura:
            logger.debug(f"[{stem}] Fragmento descartado: {razon}")
            continue
            
        page = pdf_doc[page_no - 1]
        bbox_list = [cluster["bbox"].l, cluster["bbox"].t, cluster["bbox"].r, cluster["bbox"].b]
        bbox_exp = _expandir_bbox_con_textos(bbox_list, page)
        
        l = max(0.0, bbox_exp[0] - CROP_PADDING_PT)
        t = max(0.0, bbox_exp[1] - CROP_PADDING_PT)
        r = min(page.rect.width, bbox_exp[2] + CROP_PADDING_PT)
        b = min(page.rect.height, bbox_exp[3] + CROP_PADDING_PT)
        
        clip = fitz.Rect(l, t, r, b)
        mat = fitz.Matrix(RENDER_DPI / 72.0, RENDER_DPI / 72.0)
        pix = page.get_pixmap(matrix=mat, clip=clip, alpha=False)
        
        crop_filename = f"{stem}__fig{fig_idx}.png"
        crop_filepath = out_crops_dir / crop_filename
        pix.save(str(crop_filepath))
        
        figuras_validas.append({
            "id": crop_filename,
            "path": f"crops/{crop_filename}"
        })
        fig_idx += 1
        
    pdf_doc.close()
    logger.info(f"[{stem}] {len(figuras_validas)} figuras lógicas reconstruidas (desde {len(raws)} fragmentos).")

    # 4. Conciliar el Markdown
    if md_path.exists():
        md_content = md_path.read_text(encoding="utf-8")
        
        # Inyectar las imágenes válidas en los primeros placeholders
        for fig in figuras_validas:
            md_link = f"\n![Figura: {fig['id']}]({fig['path']})\n"
            md_content = md_content.replace("", md_link, 1)
            
        # Purgar los placeholders sobrantes (fragmentos agrupados o logos borrados)
        md_content = md_content.replace("", "")
        
        md_path.write_text(md_content, encoding="utf-8")
        logger.info(f"✅ [{stem}] Markdown conciliado y purgado de basura.")

def main():
    parser = argparse.ArgumentParser(description="Etapa 2: Reconstrucción Visual Híbrida")
    parser.add_argument("--pdf", required=True, help="PDF original")
    parser.add_argument("--json", required=True, help="JSON de la Etapa 1")
    parser.add_argument("--md", required=True, help="Markdown de la Etapa 1")
    args = parser.parse_args()
    
    reconstruir_figuras_y_markdown(Path(args.pdf), Path(args.json), Path(args.md))

if __name__ == "__main__":
    main()