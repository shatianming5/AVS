from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF

from shared.schemas import TextBlock


CACHE_VERSION = 3


@dataclass(frozen=True)
class BlocksCache:
    pdf_id: str
    num_pages: int
    page_sizes: dict[int, tuple[float, float]]  # page_index -> (width,height) in points
    blocks: list[TextBlock]
    ocr_used: bool
    ocr_pages: list[int]


_MEM_CACHE: dict[str, BlocksCache] = {}


_RE_CAPTION = re.compile(r"^(figure|fig\.?|table|tab\.?)\s*\d+", re.IGNORECASE)
_RE_HEADING_NUM = re.compile(r"^\s*\d+(?:\.\d+)*\s+\S+")
_RE_REFERENCE = re.compile(r"^\s*references\s*$", re.IGNORECASE)


def _classify_block(text: str) -> str:
    t = text.strip()
    if not t:
        return "other"
    if _RE_CAPTION.match(t):
        return "caption"
    if _RE_REFERENCE.match(t):
        return "heading"
    if len(t) <= 80 and (_RE_HEADING_NUM.match(t) or t.isupper()):
        return "heading"
    # Heuristic: very short single-line blocks are likely headings, but we keep conservative
    if len(t) <= 40 and ("\n" not in t) and t[:1].isupper() and t.endswith((".", ":")) is False:
        return "heading"
    return "paragraph"


def _cache_path(pdf_id: str, data_dir: Path) -> Path:
    return data_dir / "blocks" / f"{pdf_id}.json"


def load_blocks_cache(pdf_id: str, data_dir: Path) -> BlocksCache:
    if pdf_id in _MEM_CACHE:
        return _MEM_CACHE[pdf_id]

    path = _cache_path(pdf_id, data_dir)
    if not path.exists():
        raise FileNotFoundError(path)

    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("version") != CACHE_VERSION:
        raise ValueError(f"Unsupported blocks cache version: {raw.get('version')}")

    page_sizes = {int(k): (float(v[0]), float(v[1])) for k, v in raw["page_sizes"].items()}
    blocks = [TextBlock.model_validate(b) for b in raw["blocks"]]
    ocr = raw.get("ocr") or {}
    cache = BlocksCache(
        pdf_id=pdf_id,
        num_pages=int(raw["num_pages"]),
        page_sizes=page_sizes,
        blocks=blocks,
        ocr_used=bool(ocr.get("used", False)),
        ocr_pages=[int(x) for x in (ocr.get("pages") or [])],
    )
    _MEM_CACHE[pdf_id] = cache
    return cache


def get_page_size(pdf_id: str, page_index: int, data_dir: Path) -> tuple[float, float]:
    cache = load_blocks_cache(pdf_id, data_dir)
    if page_index not in cache.page_sizes:
        raise ValueError(f"Missing page size for {pdf_id} page {page_index}")
    return cache.page_sizes[page_index]


def was_ocr_used(pdf_id: str, data_dir: Path) -> bool:
    try:
        return load_blocks_cache(pdf_id, data_dir).ocr_used
    except (FileNotFoundError, ValueError):
        return False


def extract_blocks_cached(pdf_id: str, data_dir: Path) -> list[TextBlock]:
    try:
        return [b.model_copy(deep=True) for b in load_blocks_cache(pdf_id, data_dir).blocks]
    except (FileNotFoundError, ValueError):
        pass

    pdf_path = data_dir / "pdfs" / f"{pdf_id}.pdf"
    doc = fitz.open(pdf_path)
    try:
        blocks: list[TextBlock] = []
        page_sizes: dict[int, tuple[float, float]] = {}
        for page_i in range(doc.page_count):
            page = doc.load_page(page_i)
            page_index = page_i + 1
            page_sizes[page_index] = (float(page.rect.width), float(page.rect.height))

            for bi, b in enumerate(page.get_text("blocks")):
                x0, y0, x1, y1, text, *_rest = b
                raw_text = str(text).strip()
                clean_text = re.sub(r"[ \\t]+", " ", raw_text).strip()
                if not clean_text:
                    continue

                block_type = _classify_block(clean_text)
                block_id = f"{pdf_id}:{page_index}:{bi}"
                blocks.append(
                    TextBlock(
                        block_id=block_id,
                        pdf_id=pdf_id,
                        page_index=page_index,
                        text=clean_text,
                        raw_text=raw_text,
                        clean_text=clean_text,
                        bbox=[float(x0), float(y0), float(x1), float(y1)],
                        block_type=block_type,  # type: ignore[arg-type]
                        section_path=[],
                    )
                )

        blocks = _merge_caption_blocks(blocks)

        # OCR fallback if text extraction is likely empty (e.g. scanned PDFs).
        paragraph_blocks = [b for b in blocks if b.block_type == "paragraph"]
        total_chars = sum(len((b.text or "").strip()) for b in blocks)
        ocr_used = False
        ocr_pages: list[int] = []
        # OCR is expensive; only trigger when extracted text looks truly empty/invalid.
        if (len(paragraph_blocks) < 2 and total_chars < 50) and doc.page_count > 0:
            from worker.pipeline.ocr import ocr_pages_to_blocks

            pages = list(range(1, min(doc.page_count, 6) + 1))
            ocr_res = ocr_pages_to_blocks(pdf_id=pdf_id, pdf_path=pdf_path, page_indices=pages, data_dir=data_dir, zoom=2.0)
            if ocr_res.used:
                ocr_used = True
                ocr_pages = pages
                blocks = _merge_caption_blocks(ocr_res.blocks)

        out = {
            "version": CACHE_VERSION,
            "pdf_id": pdf_id,
            "num_pages": doc.page_count,
            "page_sizes": {str(k): [v[0], v[1]] for k, v in page_sizes.items()},
            "ocr": {"used": ocr_used, "pages": ocr_pages, "zoom": 2.0},
            "blocks": [b.model_dump(mode="json") for b in blocks],
        }
        path = _cache_path(pdf_id, data_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

        cache = BlocksCache(
            pdf_id=pdf_id,
            num_pages=doc.page_count,
            page_sizes=page_sizes,
            blocks=blocks,
            ocr_used=ocr_used,
            ocr_pages=ocr_pages,
        )
        _MEM_CACHE[pdf_id] = cache
        return [b.model_copy(deep=True) for b in blocks]
    finally:
        doc.close()


def _merge_caption_blocks(blocks: list[TextBlock]) -> list[TextBlock]:
    # Merge multi-block captions into one TextBlock with bbox_list (better evidence highlighting).
    if not blocks:
        return blocks

    def bbox_union(bbs: list[list[float]]) -> list[float]:
        xs0 = [bb[0] for bb in bbs]
        ys0 = [bb[1] for bb in bbs]
        xs1 = [bb[2] for bb in bbs]
        ys1 = [bb[3] for bb in bbs]
        return [float(min(xs0)), float(min(ys0)), float(max(xs1)), float(max(ys1))]

    out: list[TextBlock] = []
    by_page: dict[int, list[TextBlock]] = {}
    for b in blocks:
        by_page.setdefault(int(b.page_index), []).append(b)

    for page_index in sorted(by_page.keys()):
        page_blocks = by_page[page_index]
        page_blocks.sort(key=lambda b: (float(b.bbox[1]) if b.bbox else 0.0, float(b.bbox[0]) if b.bbox else 0.0, b.block_id))

        i = 0
        while i < len(page_blocks):
            b = page_blocks[i]
            if b.block_type != "caption" or not b.bbox:
                out.append(b)
                i += 1
                continue

            parts = [b]
            j = i + 1
            while j < len(page_blocks):
                nxt = page_blocks[j]
                if not nxt.bbox:
                    break
                if nxt.block_type == "heading":
                    break
                if _RE_CAPTION.match((nxt.text or "").strip()):
                    break

                gap = float(nxt.bbox[1]) - float(parts[-1].bbox[3])  # type: ignore[index]
                if gap > 12.0:
                    break
                if gap < -4.0:
                    break

                x0_delta = abs(float(nxt.bbox[0]) - float(parts[0].bbox[0]))  # type: ignore[index]
                if x0_delta > 30.0:
                    break

                # Keep conservative: avoid swallowing full paragraphs.
                if len((nxt.text or "").strip()) > 240:
                    break

                parts.append(nxt)
                j += 1
                if len(parts) >= 6:
                    break

            if len(parts) == 1:
                out.append(b)
                i += 1
                continue

            bbs = [p.bbox for p in parts if p.bbox is not None]  # type: ignore[assignment]
            merged = TextBlock.model_validate(b.model_dump(mode="json"))
            merged.block_type = "caption"
            merged.bbox_list = [list(bb) for bb in bbs]
            merged.bbox = bbox_union([list(bb) for bb in bbs])
            merged_raw = "\n".join((p.raw_text or p.text or "").strip() for p in parts if (p.raw_text or p.text))
            merged_text = " ".join((p.text or "").strip() for p in parts if p.text).strip()
            merged.text = merged_text
            merged.raw_text = merged_raw or merged_text
            merged.clean_text = merged_text
            out.append(merged)
            i = j

    return out
