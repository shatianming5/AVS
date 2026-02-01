from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import fitz


@dataclass(frozen=True)
class DemoSpec:
    paper_index: int
    field: str
    venue: str
    mentor_name: str
    connector: str
    hedge: str
    method_name: str
    task_name: str


def _a4_page(doc: fitz.Document) -> fitz.Page:
    # A4 in points
    return doc.new_page(width=595, height=842)


def _tb(page: fitz.Page, x0: float, y0: float, x1: float, y1: float, text: str, *, fontsize: int) -> None:
    page.insert_textbox(fitz.Rect(x0, y0, x1, y1), text, fontsize=fontsize)


def _make_one_pdf(*, out_path: Path, spec: DemoSpec) -> None:
    doc = fitz.open()
    try:
        # Page 1: title + abstract + intro
        p1 = _a4_page(doc)
        title = f"{spec.method_name}: Mentor-Style Robust {spec.task_name} for {spec.field}"
        _tb(p1, 50, 40, 545, 70, title, fontsize=16)
        _tb(
            p1,
            50,
            78,
            545,
            128,
            "Abstract\nWe study a practical problem in modern ML systems. We show how mentor-style storytelling can be extracted "
            "from a small set of reference papers and grounded with clickable evidence.",
            fontsize=11,
        )
        _tb(p1, 50, 140, 545, 162, "1 Introduction", fontsize=14)

        intro_paras = [
            # Context
            f"Recent progress in {spec.field.lower()} has enabled strong performance across a wide range of applications, "
            f"driving rapid adoption in both academia and industry. This trend is particularly visible in {spec.venue}.",
            # Problem
            f"Despite these advances, robust {spec.task_name.lower()} remains challenging when data are noisy, sparse, or "
            "shifted from the training distribution.",
            # Gap
            f"{spec.connector}, existing approaches still lack a simple, evidence-grounded recipe that connects narrative structure "
            "to actionable writing templates while maintaining strict traceability.",
            # Approach
            f"In this paper, we propose {spec.method_name}, a lightweight pipeline that aligns Introduction structure across three mentor papers "
            "and generates a blueprint, reusable paragraph templates, and a figure storyboard with page-level evidence pointers (Figure 1).",
            # Contributions
            "Our contributions are threefold: (1) a minimal MIU pipeline for parsing and labeling rhetorical moves; "
            "(2) evidence-first rule extraction that avoids hallucinated style claims; (3) a practical report UI with viewer highlights.",
            # Roadmap
            f"The rest of this paper is organized as follows: Section 2 describes {spec.method_name}; Section 3 presents experiments and ablations; "
            "Section 4 discusses limitations and future work.",
        ]
        y = 172
        for para in intro_paras:
            _tb(p1, 50, y, 545, y + 64, para, fontsize=11)
            y += 72

        # Page 2: method + multi-block caption (bbox_list)
        p2 = _a4_page(doc)
        _tb(p2, 50, 40, 545, 62, "2 Method", fontsize=14)
        _tb(
            p2,
            50,
            70,
            545,
            165,
            f"We outline {spec.method_name} in three stages: (i) extract page-level blocks with coordinates; "
            f"(ii) label rhetorical moves with {spec.hedge} conservative priors; (iii) synthesize an evidence-linked report. "
            "We avoid copying source text verbatim by enforcing slot-based templates.",
            fontsize=11,
        )
        _tb(p2, 50, 185, 545, 240, "2.1 Evidence-first design\nEach conclusion must be backed by a PDF page anchor and bbox.", fontsize=11)
        # Multi-block caption: two adjacent blocks to trigger bbox_list merge.
        cap_y = 420
        _tb(p2, 50, cap_y, 545, cap_y + 18, "Figure 1: Overview of the end-to-end SkillPack pipeline.", fontsize=10)
        _tb(
            p2,
            50,
            cap_y + 18,
            545,
            cap_y + 38,
            f"This figure illustrates parsing, labeling, and report generation in a {spec.mentor_name}-style workflow.",
            fontsize=10,
        )
        _tb(p2, 50, 470, 545, 520, "2.2 Implementation details\nCaching is used to reduce repeated token costs.", fontsize=11)

        # Page 3: experiments + table caption (also multi-block)
        p3 = _a4_page(doc)
        _tb(p3, 50, 40, 545, 62, "3 Experiments", fontsize=14)
        _tb(
            p3,
            50,
            70,
            545,
            140,
            f"We evaluate whether {spec.method_name} produces stable outputs and whether evidence coverage exceeds 0.8. "
            f"We also test whether templates remain non-plagiaristic under a simple similarity threshold.",
            fontsize=11,
        )
        tab_y = 260
        _tb(p3, 50, tab_y, 545, tab_y + 18, "Table 1: Summary of key metrics on synthetic mentor papers.", fontsize=10)
        _tb(p3, 50, tab_y + 18, 545, tab_y + 38, "We report evidence coverage, alignment strength, and cache hit rates.", fontsize=10)
        _tb(p3, 50, 320, 545, 460, "3.1 Results\nEvidence coverage is consistently high; alignment is stable across runs.", fontsize=11)

        # Page 4: limitations + references heading to test section boundaries
        p4 = _a4_page(doc)
        _tb(p4, 50, 40, 545, 62, "4 Limitations", fontsize=14)
        _tb(
            p4,
            50,
            70,
            545,
            150,
            f"{spec.connector}, the demo uses only three papers; some fields may require more references. "
            f"LLM outputs {spec.hedge} vary by provider; we mitigate this via caching and evidence-first downgrades.",
            fontsize=11,
        )
        _tb(p4, 50, 200, 545, 222, "References", fontsize=14)
        _tb(p4, 50, 230, 545, 330, "[1] Example Reference.\n[2] Another Reference.", fontsize=10)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        doc.save(str(out_path))
    finally:
        doc.close()


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate 3 demo PDFs for paper_skill UI validation")
    ap.add_argument("--out-dir", default="data/demo_pdfs")
    ap.add_argument("--field", default="NLP")
    ap.add_argument("--venue", default="NeurIPS")
    ap.add_argument("--mentor-name", default="Prof.X")
    ap.add_argument("--prefix", default="mentor_demo")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    field = str(args.field).strip() or "NLP"
    venue = str(args.venue).strip() or "NeurIPS"
    mentor = str(args.mentor_name).strip() or "Prof.X"
    prefix = str(args.prefix).strip() or "mentor_demo"

    specs = [
        DemoSpec(1, field, venue, mentor, "However", "may", "MentorPack", "Style Transfer"),
        DemoSpec(2, field, venue, mentor, "Nevertheless", "might", "EvidencePack", "Story Alignment"),
        DemoSpec(3, field, venue, mentor, "Yet", "can", "BlueprintPack", "Template Grounding"),
    ]
    out_paths: list[Path] = []
    for s in specs:
        out = out_dir / f"{prefix}_{s.paper_index}.pdf"
        _make_one_pdf(out_path=out, spec=s)
        out_paths.append(out)

    print("Generated demo PDFs:")
    for p in out_paths:
        print(f"- {p}")
    print("\nSuggested form values:")
    print(f"- Pack name: Demo Pack ({prefix})")
    print(f"- Field hint: {field}")
    print(f"- Target venue hint: {venue}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

