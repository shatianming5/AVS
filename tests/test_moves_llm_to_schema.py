from __future__ import annotations

from pathlib import Path

from shared.schemas import TextBlock


def test_moves_llm_labels_blocks(tmp_path: Path) -> None:
    from worker.pipeline.moves import label_moves_cached

    data_dir = tmp_path / "data"
    blocks = [
        TextBlock(
            block_id="pdf1:1:0",
            pdf_id="pdf1",
            page_index=1,
            text="However, existing approaches still lack robustness.",
            clean_text="However, existing approaches still lack robustness.",
            block_type="paragraph",
            section_path=["Introduction"],
        ),
        TextBlock(
            block_id="pdf1:1:1",
            pdf_id="pdf1",
            page_index=1,
            text="In this paper, we propose MethodX.",
            clean_text="In this paper, we propose MethodX.",
            block_type="paragraph",
            section_path=["Introduction"],
        ),
    ]

    moves, hit = label_moves_cached("pdf1", blocks, field_hint="NLP", data_dir=data_dir)
    assert hit is False
    assert [m.block_id for m in moves] == ["pdf1:1:0", "pdf1:1:1"]
    assert moves[0].label == "Gap"
    assert moves[1].label == "Approach"

