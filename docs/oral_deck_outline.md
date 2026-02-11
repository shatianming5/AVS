# Oral Deck Outline (Slide-by-Slide)

Goal: turn the repo artifacts into an **oral-ready** deck with minimal extra work.

This outline references committed slide assets under `docs/oral_assets/` and canonical run artifacts under `runs/`.

## Slide 1: Title + One-Line Claim

- Title: Listen-then-Look (Audio as a Cheap Temporal Index)
- One-line claim (revised claim): Under **strict equal visual-token budgets**, audio-guided candidate moments yield consistent gains with a large oracle ceiling, while exposing a measurable stage-1 reliability gap.

## Slide 2: Problem Setup (Budget Is the Contract)

- Define the budget unit: ViT visual patch tokens.
- Define the protocol: same VLM, same prompt, same split; only frame selection changes.
- Point to the sealed accounting: `docs/oral_narrative.md`

## Slide 3: Method (1 Diagram, No Math)

- Pipeline: listen (audio index) → propose anchors → allocate visual budget (low/base/high) → answer.
- Mention controls you always report: uniform, random anchors, cheap-visual anchors.

## Slide 4: Life-or-Death Figure #1 (Acc–Tok Pareto)

- Insert: `docs/oral_assets/fig1_pareto.png`
- Talking points:
  - oracle ceiling exists at fixed budgets
  - predicted remains behind oracle; can regress vs uniform at some budgets
  - random / cheap-visual do not trivially match (controls)

## Slide 5: Life-or-Death Figure #2 (Why +2% Is Hard)

- Insert: `docs/oral_assets/fig2_c0003_decomposition.png`
- Talking points:
  - oracle ceiling vs best deployable gap (stage-1 reliability)
  - dilution (fallback fraction) + harmful buckets (far anchors / 2-high regime)
  - this explains why the hard gate is not reached yet without a *new* stage-1 signal

## Slide 6: Life-or-Death Figure #3 (Robustness / Alpha Floor)

- Insert: `docs/oral_assets/fig3_degradation_delta_acc_alpha0p5.png`
- Optional backup: `docs/oral_assets/fig3_degradation_recall_d0_alpha0p5.png`
- Talking points:
  - degradation is smooth under shift/noise/silence
  - anchored never violates the computable alpha-floor (uniform fallback)

## Slide 7: Long-Video QA Add-on (Minimal But Reviewer-Proof)

- Insert (2-up if possible):
  - `docs/oral_assets/fig4_qa_budget_curve_intentqa.png`
  - `docs/oral_assets/fig4_qa_budget_curve_avqa.png`
- Talking points:
  - include `text_only` and `random` baselines (anti-cherry-pick)
  - budget curves show where audio helps and where it flips (must state negatives explicitly)

## Slide 8: Related Work Positioning (1 Table + 1 Boundary Sentence)

- 2x2 axes: query-aware selection vs token reallocation vs training-free selection vs dynamic/agentic sampling.
- Boundary sentence: controlled equal-budget frame selection, not long-video leaderboard SOTA.
- Source list: `docs/oral_related_work.md`

## Slide 9: Limitations + What Would Make +2% Happen

- Limitation: stage-1 reliability; far-anchor / 2-high harm regimes.
- Concrete future: new stage-1 signal (not energy/clipdiff/panns/ast/fused), validated by the pre-registered promotion gate.

## Slide 10: Reproducibility / Artifact Map

- “Seals”: `bash scripts/datasets/verify_all.sh`, `python scripts/plan_evidence_matrix.py --write-docs-md`, `python -m avs.smoke`
- Ledger: `docs/experiment.md`
- Evidence: `docs/evidence_matrix.md`

