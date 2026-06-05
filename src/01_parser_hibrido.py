import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import json
import logging
import argparse
from pathlib import Path
import fitz

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from docling_core.types.doc.base import ImageRefMode
from docling_core.types.doc.document import TableItem, PictureItem
from docling_core.types.doc.base import BoundingBox, CoordOrigin

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("extractor_unificado")

RENDER_DPI = 300
CROP_PADDING_PT = 5.0
TEXT_LABEL_MARGIN = 50.0  
CLUSTER_GAP_FRAC = 0.06
MIN_PERP_OVERLAP = 0.15
MIN_AREA_FRAC = 0.015       
MAX_ASPECT_RATIO = 12.0     
HEADER_FOOTER_BAND = 0.08   

def _bbox_topleft(prov, page_height: float) -> BoundingBox:
    bb = prov.bbox
    if bb.coord_origin == CoordOrigin.BOTTOMLEFT:
        bb = bb.to_top_left_origin(page_height=page_height)
    return bb

def _vertical_gap(a, b): return (b.t - a.b) if a.b <= b.t else (a.t - b.b if b.b <= a.t else 0.0)
def _horizontal_gap(a, b): return (b.l - a.r) if a.r <= b.l else (a.l - b.r if b.r <= a.l else 0.0)

def _perp_overlap_frac(a, b, axis):
    if axis == "vertical":
        lo, hi = max(a.l, b.l), min(a.r, b.r)
        span = min(a.r - a.l, b.r - b.l)
    else:
        lo, hi = max(a.t, b.t), min(a.b, b.b)
        span = min(a.b - a.t, b.b - b.t)
    return max(0.0, hi - lo) / span if span > 0 else 0.0

def _are_contiguous(a, b, page_w, page_h):
    if _vertical_gap(a, b) <= CLUSTER_GAP_FRAC * page_h and _perp_overlap_frac(a, b, "vertical") >= MIN_PERP_OVERLAP: return True
    if _horizontal_gap(a, b) <= CLUSTER_GAP_FRAC * page_w and _perp_overlap_frac(a, b, "horizontal") >= MIN_PERP_OVERLAP: return True
    return False

def _is_garbage(bbox, page_w, page_h):
    w, h = bbox.r - bbox.l, bbox.b - bbox.t
    if w <= 0 or h <= 0: return True
    area_frac = (w * h) / (page_w * page_h)
    if area_frac < MIN_AREA_FRAC: return True
    if max(w / h, h / w) > MAX_ASPECT_RATIO: return True
    if (bbox.t < HEADER_FOOTER_BAND * page_h or bbox.b > (1 - HEADER_FOOTER_BAND) * page_h) and area_frac < 0.05: return True
    return False

def _expandir_bbox_con_textos(bbox, page):
    x0, y0, x1, y1 = bbox
    pw, ph = page.rect.width, page.rect.height
    search_rect = fitz.Rect(max(0, x0 - TEXT_LABEL_MARGIN), max(0, y0 - TEXT_LABEL_MARGIN),
                            min(pw, x1 + TEXT_LABEL_MARGIN), min(ph, y1 + TEXT_LABEL_MARGIN))
    nx0, ny0, nx1, ny1 = x0, y0, x1, y1
    for b in page.get_text("dict")["blocks"]:
        if b.get("type") != 0: continue 
        bx0, by0, bx1, by1 = b["bbox"]
        if (bx1 - bx0) > pw * 0.35: continue 
        if fitz.Rect(bx0, by0, bx1, by1).intersects(search_rect):
            nx0, ny0, nx1, ny1 = min(nx0, bx0), min(ny0, by0), max(nx1, bx1), max(ny1, by1)
    return [nx0, ny0, nx1, ny1]

def _resolve_caption(item, doc) -> str:
    """Extrae el puntero duro del caption desde el cerebro de Docling."""
    parts = []
    if hasattr(item, "captions") and item.captions:
        for cap_ref in item.captions:
            try:
                node = cap_ref.resolve(doc)
                if getattr(node, "text", None):
                    parts.append(node.text.strip())
            except: pass
    return " ".join(parts).strip()

def procesar_documento_unificado(pdf_path: Path, output_dir: Path, images_scale: float = 2.0):
    stem = pdf_path.stem
    paper_out_dir = output_dir / stem
    out_crops_dir = paper_out_dir / "crops"
    out_crops_dir.mkdir(parents=True, exist_ok=True)
    
    json_path = paper_out_dir / f"{stem}.docling.json"
    manifest_path = paper_out_dir / "visual_manifest.json"
    
    logger.info(f"[{stem}] Parseo estructural con Docling...")
    pipeline_options = PdfPipelineOptions()
    pipeline_options.generate_picture_images = True
    pipeline_options.images_scale = images_scale
    pipeline_options.do_table_structure = True 
    converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)})
    doc_result = converter.convert(pdf_path)
    documento = doc_result.document
    
    raw_items = []
    for item, _ in documento.iterate_items():
        if isinstance(item, (PictureItem, TableItem)) and item.prov:
            prov = item.prov[0]
            raw_items.append({
                "page": prov.page_no,
                "bbox": _bbox_topleft(prov, documento.pages[prov.page_no].size.height),
                "caption": _resolve_caption(item, documento)
            })

    pdf_doc = fitz.open(str(pdf_path))
    by_page = {}
    for r in raw_items: by_page.setdefault(r["page"], []).append(r)
        
    clusters = []
    for page_no, items_in_page in by_page.items():
        page_w, page_h = documento.pages[page_no].size.width, documento.pages[page_no].size.height
        parent = list(range(len(items_in_page)))
        def find(x):
            while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
            return x
        def union(x, y): parent[find(x)] = find(y)
            
        for i in range(len(items_in_page)):
            for j in range(i + 1, len(items_in_page)):
                if _are_contiguous(items_in_page[i]["bbox"], items_in_page[j]["bbox"], page_w, page_h):
                    union(i, j)
                    
        groups = {}
        for idx, p in enumerate(items_in_page): groups.setdefault(find(idx), []).append(p)
            
        for group in groups.values():
            bboxes = [p["bbox"] for p in group]
            captions = [p["caption"] for p in group if p["caption"]]
            clusters.append({
                "page": page_no,
                "bbox": BoundingBox.enclosing_bbox(bboxes),
                "caption": max(captions, key=len) if captions else ""
            })

    items_validos = []
    for item_idx, cluster in enumerate(clusters):
        page_no = cluster["page"]
        page_w, page_h = documento.pages[page_no].size.width, documento.pages[page_no].size.height
        if _is_garbage(cluster["bbox"], page_w, page_h): continue
            
        page = pdf_doc[page_no - 1]
        bbox_exp = _expandir_bbox_con_textos([cluster["bbox"].l, cluster["bbox"].t, cluster["bbox"].r, cluster["bbox"].b], page)
        
        l, t = max(0.0, bbox_exp[0] - CROP_PADDING_PT), max(0.0, bbox_exp[1] - CROP_PADDING_PT)
        r, b = min(page.rect.width, bbox_exp[2] + CROP_PADDING_PT), min(page.rect.height, bbox_exp[3] + CROP_PADDING_PT)
        
        pix = page.get_pixmap(matrix=fitz.Matrix(RENDER_DPI / 72.0, RENDER_DPI / 72.0), clip=fitz.Rect(l, t, r, b), alpha=False)
        crop_filename = f"{stem}__item{item_idx}.png"
        pix.save(str(out_crops_dir / crop_filename))
        
        items_validos.append({
            "item_id": crop_filename,
            "crop_path": f"crops/{crop_filename}",
            "caption_oficial": cluster["caption"],
            "page": page_no
        })
        
    pdf_doc.close()
    documento.save_as_json(json_path, image_mode=ImageRefMode.PLACEHOLDER)
    
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(items_validos, f, indent=2, ensure_ascii=False)
        
    logger.info(f"✅ [{stem}] Éxito. Manifiesto guardado con {len(items_validos)} elementos.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()
    procesar_documento_unificado(Path(args.pdf), Path(args.out))

if __name__ == "__main__":
    main()