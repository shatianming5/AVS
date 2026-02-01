from __future__ import annotations

import csv
import subprocess
from dataclasses import dataclass
from pathlib import Path

import fitz

from shared.schemas import TextBlock


@dataclass(frozen=True)
class OcrResult:
    blocks: list[TextBlock]
    used: bool


def ocr_pages_to_blocks(
    *,
    pdf_id: str,
    pdf_path: Path,
    page_indices: list[int],
    data_dir: Path,
    zoom: float = 2.0,
    lang: str = "eng",
) -> OcrResult:
    if not page_indices:
        return OcrResult(blocks=[], used=False)

    blocks: list[TextBlock] = []
    doc = fitz.open(pdf_path)
    try:
        for page_index in page_indices:
            if page_index < 1 or page_index > doc.page_count:
                continue
            page = doc.load_page(page_index - 1)
            img_path = _render_page_cached(pdf_id=pdf_id, page_index=page_index, page=page, data_dir=data_dir, zoom=zoom)
            tsv = _run_tesseract_tsv(img_path, lang=lang)
            blocks.extend(_tsv_to_blocks(pdf_id=pdf_id, page_index=page_index, tsv_text=tsv, zoom=zoom))
        return OcrResult(blocks=blocks, used=True if blocks else False)
    finally:
        doc.close()


def _render_page_cached(*, pdf_id: str, page_index: int, page: fitz.Page, data_dir: Path, zoom: float) -> Path:
    z = str(zoom).replace(".", "_")
    out = data_dir / "cache" / "ocr_images" / pdf_id / f"p{page_index}_z{z}.png"
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    pix.save(out.as_posix())
    return out


def _run_tesseract_tsv(image_path: Path, *, lang: str) -> str:
    # tesseract <img> stdout -l eng --psm 6 tsv
    proc = subprocess.run(
        ["tesseract", image_path.as_posix(), "stdout", "-l", lang, "--psm", "6", "tsv"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"tesseract failed: {proc.stderr.strip()}")
    return proc.stdout


def _tsv_to_blocks(*, pdf_id: str, page_index: int, tsv_text: str, zoom: float) -> list[TextBlock]:
    # TSV columns: level,page_num,block_num,par_num,line_num,word_num,left,top,width,height,conf,text
    reader = csv.DictReader(tsv_text.splitlines(), delimiter="\t")
    # group words by (block_num, par_num, line_num)
    groups: dict[tuple[str, str, str], list[dict]] = {}
    for row in reader:
        if not row:
            continue
        if row.get("level") != "5":
            continue
        text = (row.get("text") or "").strip()
        if not text:
            continue
        key = (row.get("block_num") or "0", row.get("par_num") or "0", row.get("line_num") or "0")
        groups.setdefault(key, []).append(row)

    out: list[TextBlock] = []
    for idx, (_k, rows) in enumerate(sorted(groups.items(), key=lambda kv: kv[0])):
        words = []
        xs: list[float] = []
        ys: list[float] = []
        xe: list[float] = []
        ye: list[float] = []
        for r in rows:
            words.append((r.get("text") or "").strip())
            left = float(r.get("left") or 0)
            top = float(r.get("top") or 0)
            width = float(r.get("width") or 0)
            height = float(r.get("height") or 0)
            xs.append(left / zoom)
            ys.append(top / zoom)
            xe.append((left + width) / zoom)
            ye.append((top + height) / zoom)
        if not words:
            continue
        text = " ".join(words)
        bbox = [min(xs), min(ys), max(xe), max(ye)]
        block_id = f"{pdf_id}:{page_index}:ocr:{idx}"
        out.append(
            TextBlock(
                block_id=block_id,
                pdf_id=pdf_id,
                page_index=page_index,
                text=text,
                raw_text=text,
                clean_text=text,
                bbox=bbox,
                block_type="paragraph",  # type: ignore[arg-type]
                section_path=[],
            )
        )
    return out

