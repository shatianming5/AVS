# Oral Related Work Alignment (R0700)

Date: 2026-02-11

This page aligns our "listen-then-look" QA add-on with the closest public work, focusing on reviewer-facing differentiators:

- strict equal budget accounting
- audio-aware indexing
- long-video QA transfer under budget

## 1) Closest Works (with links)

### Datasets / Benchmarks

- IntentQA (ICCV 2023): https://openaccess.thecvf.com/content/ICCV2023/html/Li_IntentQA_Context-aware_Video_Intent_Reasoning_ICCV_2023_paper.html
- EgoSchema (NeurIPS D&B 2023): https://arxiv.org/abs/2308.09126
- LongVideoBench (2024): https://arxiv.org/abs/2407.15754
- AVQA (dataset repo): https://github.com/AlyssaYoung/AVQA
- MUSIC-AVQA (CVPR 2022): https://arxiv.org/abs/2203.14072

### Long-video MLLM / Frame Selection

- LongVA (2024): https://arxiv.org/abs/2406.16852
- VidF4 (2024): https://arxiv.org/abs/2407.15047
- M-LLM based Video Frame Selection (2025): https://arxiv.org/abs/2502.19680

### Audio-Text Foundation (for query-audio relevance)

- CLAP (2022): https://arxiv.org/abs/2206.04769

## 2) Positioning Matrix

| Work | Task emphasis | Long-video setting | Query-aware selection | Audio-aware selection | Strict equal budget report |
|---|---|---|---|---|---|
| LongVA | Long video understanding | Yes | Partial (model-side) | No (not explicit) | Not its main focus |
| VidF4 | Efficient video understanding | Yes | Yes | No (visual-centric) | Yes (efficiency-oriented) |
| M-LLM frame selection | Frame selection policy | Yes | Yes | No (mainly visual/text) | Varies by setup |
| IntentQA / EgoSchema / AVQA | QA benchmarks | Yes | N/A | N/A | N/A |
| **Ours (QA add-on)** | Budgeted QA transfer | Yes | Yes (`ql2l_*`) | **Yes (`ql2l_clap`, `ql2l_asr_bm25`)** | **Yes (B_FRAMES + same VLM)** |

## 3) Reviewer-facing claim boundary

- We do **not** claim SOTA on the full long-video QA leaderboards.
- We claim a **controlled budgeted transfer result**:
  - same VLM, same prompt style, same evaluation split,
  - only frame selection differs,
  - and we include anti-cherry-pick controls (`random`, `text_only`, and negative baselines).

## 4) How this maps to current artifacts

- IntentQA / EgoSchema / AVQA core runs: `docs/experiment.md` entries `E0600~E0619`
- Bucket narrative (when/where helps): `runs/E0619_qa_bucket_report_20260211-062907/*/bucket_report.md`
- Next-step significance layer (planned E0705): `runs/E0705_qa_bucket_significance_*/`

