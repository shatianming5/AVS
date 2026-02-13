# Oral Related Work Alignment (R0700)

Date: 2026-02-14

This page aligns our "listen-then-look" QA add-on with the closest public work, focusing on reviewer-facing differentiators:

- strict equal budget accounting
- audio-aware indexing
- long-video QA transfer under budget

## 1) Closest Works (with links)

### Datasets / Benchmarks

- IntentQA (ICCV 2023): https://openaccess.thecvf.com/content/ICCV2023/html/Li_IntentQA_Context-aware_Video_Intent_Reasoning_ICCV_2023_paper.html
- EgoSchema (NeurIPS D&B 2023): https://arxiv.org/abs/2308.09126
- LongVideoBench (2024): https://arxiv.org/abs/2407.15754
- Video-MME (2024): https://arxiv.org/abs/2405.21075
- AVQA (dataset repo): https://github.com/AlyssaYoung/AVQA
- MUSIC-AVQA (CVPR 2022): https://arxiv.org/abs/2203.14072

### Audio-Visual Event Localization (AVE / temporal grounding)

- AVE (ECCV 2018) baseline code: https://github.com/YapengTian/AVE-ECCV18
- AVE-CLIP (2022): https://arxiv.org/abs/2210.05060
- CPSP / PSP (2022; contrastive positive sample propagation for AVE localization): https://arxiv.org/abs/2211.09980
- CACE-Net (2024): https://arxiv.org/abs/2408.01952
- UniAV (2024): https://arxiv.org/abs/2404.03179
- Towards Open-Vocabulary Audio-Visual Event Localization (2024): https://arxiv.org/abs/2411.11278

### Long-video MLLM / Frame Selection

- LongVA (2024): https://arxiv.org/abs/2406.16852
- LongVU (2024): https://arxiv.org/abs/2410.17434
- Logic-in-Frames (2025; dynamic keyframe search via semantic-logical verification): https://arxiv.org/abs/2503.13139
- VidF4 (2024): https://arxiv.org/abs/2407.15047
- Flow4Agent (2025; motion-prior long-form video agent): https://arxiv.org/abs/2510.05836
- VSI (2025; visual-subtitle integration keyframe selection): https://arxiv.org/abs/2508.06869
- Efficient Video Sampling (EVS) (2025): https://arxiv.org/abs/2510.14624
- M-LLM based Video Frame Selection (2025): https://arxiv.org/abs/2502.19680
- Adaptive Keyframe Sampling (CVPR 2025): https://openaccess.thecvf.com/content/CVPR2025/html/Tang_Adaptive_Keyframe_Sampling_for_Long_Video_Understanding_CVPR_2025_paper.html
- FrameMind (2025; RL dynamic sampling during reasoning): https://arxiv.org/abs/2509.24008
- MaxInfo (2025; training-free max-volume frame selection): https://arxiv.org/abs/2502.03183
- Q-Frame (ICCV 2025; query-aware frame selection + multi-resolution adaptation): https://openaccess.thecvf.com/content/ICCV2025/html/Riaz_Q-Frame_An_Efficient_Approach_to_Query-aware_Video_Frame_Selection_ICCV_2025_paper.html
- AdaRD-Key (2025; relevance-diversity keyframe selection): https://arxiv.org/abs/2510.02778
- FOCUS (2025; training-free keyframe selection via bandits): https://arxiv.org/abs/2510.27280
- DIG (2025; dynamically balances global vs local selection by query type): https://arxiv.org/abs/2512.04000
- Video-in-the-Loop (2025; localize-then-answer with token reallocation under a fixed budget): https://arxiv.org/abs/2510.04022
- VideoBrain (2026; adaptive frame sampling / multi-agent long-video understanding): https://arxiv.org/abs/2602.04094
- Adaptive Video Understanding Agent (2024; long-form agentic pipeline): https://arxiv.org/abs/2410.20252
- TiFRe (2026; text-guided frame reduction for video MLLMs): https://arxiv.org/abs/2602.08861

### Audio-Text Foundation (for query-audio relevance)

- CLAP (2022): https://arxiv.org/abs/2206.04769

## 2) Positioning Matrix

| Work | Task emphasis | Long-video setting | Query-aware selection | Audio-aware selection | Strict equal budget report |
|---|---|---|---|---|---|
| LongVA | Long video understanding | Yes | Partial (model-side) | No (not explicit) | Not its main focus |
| VidF4 | Efficient video understanding | Yes | Yes | No (visual-centric) | Yes (efficiency-oriented) |
| M-LLM frame selection | Frame selection policy | Yes | Yes | No (mainly visual/text) | Varies by setup |
| Adaptive Keyframe Sampling | Efficient long video understanding | Yes | Varies by task | No | Efficiency-focused |
| FrameMind | Dynamic sampling during reasoning | Yes | Yes | No | Varies by setup |
| Q-Frame | Query-aware frame selection | Yes | Yes | No | Efficiency-focused (not our exact token-equal protocol) |
| DIG | Query-type adaptive selection | Yes | Yes | No | Varies by setup |
| FOCUS | Training-free keyframe selection | Yes | Yes | No | Varies by setup |
| AdaRD-Key | Relevance-diversity selection | Yes | Yes | No | Varies by setup |
| Video-in-the-Loop | Span localization + reallocation | Yes | Yes | No | Often fixed-budget (token reallocation) |
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
- Bucket significance: `runs/E0705_qa_bucket_significance_*/bucket_significance.json`
