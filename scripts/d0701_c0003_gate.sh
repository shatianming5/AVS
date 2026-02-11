#!/usr/bin/env bash
set -euo pipefail

# D0701: C0003 gate decision (pre-registered no-infinite-search rule).

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TS="${TS:-$(date +%Y%m%d-%H%M%S)}"
OUT_DIR="${OUT_DIR:-runs/D0701_c0003_gate_${TS}}"
mkdir -p "$OUT_DIR"

TARGET_DELTA="${TARGET_DELTA:-0.02}"
TARGET_P="${TARGET_P:-0.05}"
QUICK_MIN_DELTA="${QUICK_MIN_DELTA:-0.01}"

# Candidate A: higher mean on full test402.
python -m avs.experiments.c0003_gate_decision \
  --val-summary runs/E0624_ave_p0_sweep_official_val_av_clipdiff_vec_mlp_ltl_adaptive_keepadj_v1_20260210-224555/sweep_summary.json \
  --quick-metrics runs/E0637_quick_test402_vecmlp_keepadj_adj2_shift1_std0p55_df5_officialids_20260211-000915/metrics.json \
  --full-metrics runs/E0638_full_test402_vecmlp_keepadj_adj2_shift1_std0p55_df5_officialids_s0-9_20260211-001009/metrics.json \
  --target-delta "$TARGET_DELTA" \
  --target-p "$TARGET_P" \
  --quick-min-delta "$QUICK_MIN_DELTA" \
  --out-dir "$OUT_DIR/candidate_df5"

# Candidate B: better p-value on full test402.
python -m avs.experiments.c0003_gate_decision \
  --val-summary runs/E0624_ave_p0_sweep_official_val_av_clipdiff_vec_mlp_ltl_adaptive_keepadj_v1_20260210-224555/sweep_summary.json \
  --quick-metrics runs/E0636_quick_test402_vecmlp_keepadj_adj2_shift1_std0p55_df7_officialids_20260211-000822/metrics.json \
  --full-metrics runs/E0643_full_test402_vecmlp_keepadj_adj2_shift1_std0p55_df7_officialids_s0-9_20260211-001604/metrics.json \
  --target-delta "$TARGET_DELTA" \
  --target-p "$TARGET_P" \
  --quick-min-delta "$QUICK_MIN_DELTA" \
  --out-dir "$OUT_DIR/candidate_df7"

OUT_DIR="$OUT_DIR" python - <<'PY'
import json
import os
from pathlib import Path

out_dir = Path(os.environ["OUT_DIR"])
a = json.loads((out_dir / "candidate_df5" / "decision.json").read_text())
b = json.loads((out_dir / "candidate_df7" / "decision.json").read_text())

def score(x):
    # Prioritize full_pass, then full delta, then lower p.
    obs = x["observed"]
    gr = x["gate_results"]
    return (
        1 if gr["full_pass"] else 0,
        float(obs["full_delta"]),
        -float(obs["full_p"]),
    )

best = a if score(a) >= score(b) else b
summary = {
    "ok": True,
    "latest_dir": str(out_dir),
    "candidates": {
        "df5": {
            "full_delta": a["observed"]["full_delta"],
            "full_p": a["observed"]["full_p"],
            "decision": a["gate_results"]["decision"],
        },
        "df7": {
            "full_delta": b["observed"]["full_delta"],
            "full_p": b["observed"]["full_p"],
            "decision": b["gate_results"]["decision"],
        },
    },
    "selected": {
        "full_delta": best["observed"]["full_delta"],
        "full_p": best["observed"]["full_p"],
        "decision": best["gate_results"]["decision"],
        "c0003_proven": best["gate_results"]["c0003_proven"],
    },
}
(out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(out_dir / "summary.json")
PY

echo "$OUT_DIR"
