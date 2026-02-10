# Oral Narrative Pack (Listen-then-Look / AVE-P0)

This document is the **oral-ready story skeleton** that maps directly to the experiment ledger (`docs/experiment.md`) and the actionable claims (`docs/plan.md`).

## 1) What Is New (One Sentence)

**Audio is a cheap temporal index**: we “listen” to propose a small set of candidate moments, then **allocate the visual token budget only where evidence is likely**, instead of uniformly spending compute across time.

## 2) Budget Definition Is Sealed (No Wiggle Room)

We evaluate under **strict equal visual-token budgets**.

- Unit: **ViT patch tokens** fed to the visual encoder.
- For AVE (10s clips): fixed budget points (see the multi-budget grid below).
- For EPIC-SOUNDS (up to 120s): budget defined as `max_steps × base_res` (token-equivalent) with a deterministic equal-budget low/base/high plan for anchored methods.
- Accounting artifacts:
  - Budgeted Pareto report + plot: `runs/E0330_full_av_clipdiff_mlp_auto_20260205-184559/pareto_report.json`, `runs/E0330_full_av_clipdiff_mlp_auto_20260205-184559/pareto.png`
  - Vision efficiency calibration (tokens↔latency reference): `runs/E0408_vision_efficiency_20260206-161610/vision_efficiency.json`

## 3) Pre-Registered MDE Protocol (Oracle → Predicted + Controls)

For each fixed budget, we always compare the same method family:

1. **Oracle anchors** (upper bound at the same budget)
2. **Predicted anchors** (deployable Stage-1)
3. Controls:
   - **Uniform** (spend budget evenly)
   - **Random anchors** (tests “any window works?”)
   - **Cheap-visual anchors** (tests “audio is not special?”)

Deliverable that packages this into one “生死图”:
- MDE Pareto grid: `runs/E0330_full_av_clipdiff_mlp_auto_20260205-184559/pareto.png`
- Oracle→Predicted gap grid (dense-stride wrapper): `runs/E0504_oracle_pred_gap_grid_dense_stride_full_20260207-155721/pareto_report.json`

## 4) Main Mechanism Evidence (What Reviewers Must Believe)

**(i) There is a strong ceiling at fixed budget (Oracle).**  
**(ii) Predicted closes part of the gap but remains behind Oracle (Stage-1 reliability gap).**  
**(iii) Random/cheap-visual do not trivially match predicted (controls).**

Key evidence pointers:
- Oracle vs Predicted report (token_budget=1960, official AVE test402): `runs/E0201_full_energy_20260203-210017/oracle_vs_predicted.json`
- Dense-stride Oracle→Predicted multi-budget grid (SEEDS=0..9): `runs/E0504_oracle_pred_gap_grid_dense_stride_full_20260207-155721/pareto_report.json`

## 5) Why It Works / When It Fails (Bucketed Diagnosis + Evidence Alignment)

We explicitly show **where gains come from** and **where they disappear**.

- Bucketed diagnosis identifies failure modes:
  - fallback-heavy clips (reliability gate forces uniform)
  - far-anchor plans (anchors too far from evidence)
  - “2-high harms context” regime (too many high-res anchors can remove context)
- Evidence Alignment (Coverage@τ) is reported as a **diagnostic**:
  - It is **weakly correlated** with downstream accuracy deltas in our current setup (negative-but-clean evidence).

Evidence pointers:
- Evidence Alignment report (energy baseline; test402): `runs/E0202_evidence_alignment_energy_test402_20260209-061145/evidence_alignment.json`
- Evidence Alignment report (best promoted candidate; still weak corr): `runs/E0411_evidence_alignment_av_clipdiff_flow_mlp_stride_top1med_thr0p5_20260206-182007/evidence_alignment.json`

## 6) Robustness (Degradation Curves + Alpha Lower Bound)

We pre-register and execute a **degradation protocol** (shift/noise/silence × α) and require:

- Anchor quality degrades smoothly under perturbations.
- Downstream anchored accuracy stays **no worse than the computable α-floor** (uniform fallback baseline).

Evidence pointers:
- Anchor-quality degradation suite (Recall@K,Δ grid): `runs/E0203_degradation_energy_20260209-061156/degradation_suite.json`
- Downstream degradation-accuracy grid + alpha floor checks (dense-stride wrapper; SEEDS=0..9): `runs/E0505_degradation_accuracy_dense_stride_full_20260207-161213/degradation_accuracy.json`

## 7) Cross-Dataset Proxy (Long Video)

We verify the “listen-then-look” indexing story in a long-video downstream task:

- EPIC-SOUNDS (expanded t256/v137, strict equal-budget; SEEDS=0,1,2):
  - Anchored: `runs/E0100_epic_video_cls_local_audio_anchored_full_ms120_t256_v137_s012_20260209-235834/metrics.json`
  - Uniform: `runs/E0100_epic_video_cls_local_uniform_full_ms120_t256_v137_s012_20260210-001346/metrics.json`
  - Random: `runs/E0100_epic_video_cls_local_random_full_ms120_t256_v137_s012_20260210-001929/metrics.json`

Optional diagnostic (tighter budget, `max_steps=60`): see `docs/experiment.md` E0100 notes.

## 8) Reproducibility (What We Provide)

We maintain:

- Runnable commands + required artifacts in `docs/experiment.md`
- A conclusion→evidence matrix in `docs/evidence_matrix.md`
- Per-experiment logs under `artifacts/experiments/*/run.log`

Dataset verification is captured by:
- `python scripts/datasets/verify_all.py` (outputs `runs/datasets_verify_*/datasets_verify.json`)

