# Oral Slide Assets (Committed Exports)

These are **slide-ready** exports copied/generated from the canonical run artifacts under `runs/`.

Regeneration sources are documented below so the exports are reproducible even if run IDs change.

## Figures

- `fig1_pareto.png`
  - Source: `runs/E0330_mde_pareto_grid_official_av_clipdiff_mlp_local_20260209-235305/pareto.png`

- `fig2_c0003_decomposition.png`
  - Source script: `scripts/oral/make_c0003_decomposition_fig.py`
  - Inputs:
    - Energy oracle/uniform: `runs/REAL_AVE_OFFICIAL_RERUN_20260209-054402/p0_train3339_test402_energy_160_224_352_k2_shift1_std1.0_temporal_conv/metrics.json`
    - Best config metrics/diagnose: `runs/E0643_full_test402_vecmlp_keepadj_adj2_shift1_std0p55_df7_officialids_s0-9_20260211-001604/{metrics.json,diagnose.json}`
    - Evidence Alignment: `runs/E0720_evidence_alignment_df7_best_20260212-015616/evidence_alignment.json`
  - Command:
    - `scripts/oral/make_c0003_decomposition_fig.py --energy-metrics ... --best-metrics ... --best-diagnose ... --best-evidence-alignment ... --out docs/oral_assets/fig2_c0003_decomposition.png`

- `fig3_degradation_delta_acc_alpha0p5.png`
- `fig3_degradation_recall_d0_alpha0p5.png`
  - Source: `runs/E0331_degradation_accuracy_av_clipdiff_mlp_local_20260209-235316/degradation_plots/*`

- `fig4_qa_budget_curve_avqa.png`
  - Source: `runs/E0702_qa_budget_curve_20260211-164607/avqa_curve/budget_curve.png`

- `fig4_qa_budget_curve_intentqa.png`
  - Source: `runs/E0702_qa_budget_curve_20260211-164607/intentqa_curve/budget_curve.png`
