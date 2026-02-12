# Evidence Matrix

- Generated: `2026-02-12 12:00:23`
- Plan: `docs/plan.md`

| Conclusion | Checked in plan | Local artifacts present? | Notes |
| --- | --- | --- | --- |
| `C0001` | yes | yes |  |
| `C0002` | yes | yes |  |
| `C0003` | yes | yes |  |
| `C0004` | yes | yes |  |
| `C0005` | yes | yes |  |
| `C0006` | yes | yes |  |
| `C0007` | yes | yes |  |
| `C0008` | yes | yes |  |
| `C0009` | yes | yes |  |

## C0001: On AVE, under a strictly equal ViT token budget, Audio-anchored sampling (with a lightweight temporal head) improves segment-level accuracy vs Uniform sampling and Random anchors.

- All referenced artifacts exist locally.

## C0002: Audio-based anchors achieve higher Recall@K (and Recall@K,Δ) than random anchors on AVE.

- All referenced artifacts exist locally.

## C0003: On official AVE test402, energy-based sampling-only anchored_top2 yields a modest gain over uniform (~+0.40% abs) and does not reach +2% under this config.

- All referenced artifacts exist locally.

## C0004: On official AVE test402, `audio_concat_anchored_top2` does not improve over `audio_concat_uniform` under this config.

- All referenced artifacts exist locally.

## C0005: On EPIC-SOUNDS video-level multi-label recognition, audio-anchored selection improves mAP on val (SEEDS>=3).

- All referenced artifacts exist locally.

## C0006: Oracle anchors provide a strong accuracy upper bound at the same token budget (1960) on official AVE.

- All referenced artifacts exist locally.

## C0007: Predicted anchors are still ~3-4% behind Oracle anchors at token_budget=1960 under this deployable Stage-1 config.

- All referenced artifacts exist locally.

## C0008: Evidence Alignment (Coverage@τ) has weak correlation with downstream accuracy deltas under energy anchors on test402 (diagnostic, not predictive).

- All referenced artifacts exist locally.

## C0009: Stage-1 anchor quality degrades smoothly under shift/noise/silence on official AVE test402 for energy anchors (Recall@K,Δ).

- All referenced artifacts exist locally.

