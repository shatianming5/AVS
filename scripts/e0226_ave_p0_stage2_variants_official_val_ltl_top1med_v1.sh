#!/usr/bin/env bash
set -euo pipefail

# E0226: Stage-2 plan variants on official AVE val402 for the fixed Stage-1 gate from E0223:
#   conf_metric=top1_med, thr=0.6, shift=1
#
# This script evaluates a small Stage-2 variant set by reusing prebuilt config JSONs derived from
# the latest E0223 sweep directory:
#   - best_config.json               (baseline)
#   - best_config_mixedAlloc.json    (base alloc mixed)
#   - best_config_scoreAlloc.json    (base alloc score)
#   - best_config_window3.json       (window_topk selection)
#   - best_config_nmsR2.json         (nms selection)
#
# Outputs under OUT_DIR:
#   - <variant_name>/metrics.json (+ run.log)
#   - variants_summary.json
#   - best_config.json (copy of the best-by-val-Δ config)
#   - eventness_scores.json (copied from E0223 for reuse by E0227)
#
# Usage:
#   SEEDS=0,1,2 GPUS=0,1,2,3 EVENTNESS=av_clipdiff_mlp bash scripts/e0226_ave_p0_stage2_variants_official_val_ltl_top1med_v1.sh
#
# Optional:
#   E0223_DIR=... (override which E0223 sweep dir to read configs/scores from)

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

EVENTNESS="${EVENTNESS:-av_clipdiff_mlp}"
SEEDS="${SEEDS:-0,1,2}"
GPUS="${GPUS:-0,1,2,3}"

if [[ -z "${E0223_DIR:-}" ]]; then
  E0223_DIR="$(ls -td runs/E0223_* 2>/dev/null | head -n 1 || true)"
fi
if [[ -z "${E0223_DIR}" || ! -d "${E0223_DIR}" ]]; then
  echo "ERROR: Could not find E0223_DIR (expected runs/E0223_*) and E0223_DIR not set." >&2
  exit 2
fi

OUT_DIR="${OUT_DIR:-runs/E0226_ave_p0_stage2_variants_official_val_${EVENTNESS}_$(date +%Y%m%d-%H%M%S)}"
mkdir -p "${OUT_DIR}"
export OUT_DIR

SCORES_SRC="${E0223_DIR}/eventness_scores.json"
if [[ -f "${SCORES_SRC}" ]]; then
  cp -f "${SCORES_SRC}" "${OUT_DIR}/eventness_scores.json"
else
  echo "WARNING: missing ${SCORES_SRC}; per-variant runs may recompute Stage-1 scores." >&2
fi

declare -a CONFIGS=(
  "best_config.json"
  "best_config_mixedAlloc.json"
  "best_config_scoreAlloc.json"
  "best_config_window3.json"
  "best_config_nmsR2.json"
)

declare -a GPU_ARR=()
if [[ -n "${GPUS}" ]]; then
  IFS=, read -ra GPU_ARR <<< "${GPUS}"
fi

declare -a PIDS=()
declare -a VARIANTS=()

for i in "${!CONFIGS[@]}"; do
  cfg="${CONFIGS[$i]}"
  cfg_path="${E0223_DIR}/${cfg}"
  if [[ ! -f "${cfg_path}" ]]; then
    echo "WARNING: missing ${cfg_path}; skipping." >&2
    continue
  fi

  variant="${cfg%.json}"
  out_variant="${OUT_DIR}/${variant}"
  mkdir -p "${out_variant}"
  cp -f "${cfg_path}" "${out_variant}/config.json"
  if [[ -f "${out_variant}/metrics.json" ]]; then
    echo "[e0226] found existing ${out_variant}/metrics.json; skipping run." >&2
    continue
  fi

  # Avoid races: each run gets its own score cache copy.
  if [[ -f "${OUT_DIR}/eventness_scores.json" ]]; then
    cp -f "${OUT_DIR}/eventness_scores.json" "${out_variant}/eventness_scores.json"
    SCORES_JSON="${out_variant}/eventness_scores.json"
  else
    SCORES_JSON=""
  fi

  gpu=""
  if [[ "${#GPU_ARR[@]}" -gt 0 ]]; then
    gpu="${GPU_ARR[$((i % ${#GPU_ARR[@]}))]}"
  fi

  echo "[e0226] launching variant=${variant} gpu=${gpu:-<unset>} cfg=${cfg_path}" >&2
  VARIANTS+=("${variant}")

  (
    set -euo pipefail
    if [[ -n "${gpu}" ]]; then
      export CUDA_VISIBLE_DEVICES="${gpu}"
    fi
    export EVENTNESS
    export BEST_CONFIG_JSON="${cfg_path}"
    export OUT_DIR="${out_variant}"
    export SPLIT_EVAL="val"
    export EVAL_IDS_FILE="data/AVE/meta/download_ok_val_official.txt"
    export LIMIT_EVAL="${LIMIT_EVAL:-402}"
    export SEEDS
    if [[ -n "${SCORES_JSON}" ]]; then
      export SCORES_JSON
    fi
    bash scripts/e0208_ave_p0_best_to_test_official_ltl_stage1.sh
  ) >"${out_variant}/run.log" 2>&1 &

  PIDS+=("$!")
done

for pid in "${PIDS[@]}"; do
  wait "${pid}"
done

python - <<'PY'
import json
import os
from pathlib import Path

out_dir = Path(os.environ["OUT_DIR"])
variants = []
for child in sorted(out_dir.iterdir()):
    if not child.is_dir():
        continue
    metrics = child / "metrics.json"
    if metrics.exists():
        variants.append((child.name, metrics))

rows = []
best = None
for name, metrics_path in variants:
    obj = json.loads(metrics_path.read_text(encoding="utf-8"))
    anchored = float(obj["summary"]["anchored_top2"]["mean"])
    uniform = float(obj["summary"]["uniform"]["mean"])
    delta = anchored - uniform
    p = float(obj["paired_ttest"]["anchored_vs_uniform"]["p"])
    rows.append(
        {
            "variant": name,
            "metrics_path": str(metrics_path),
            "anchored_mean": anchored,
            "uniform_mean": uniform,
            "anchored_minus_uniform_mean": delta,
            "anchored_vs_uniform_p": p,
        }
    )
    if best is None or delta > float(best["anchored_minus_uniform_mean"]):
        best = rows[-1]

payload = {
    "ok": True,
    "objective": "val402 Stage-2 variant selection for fixed top1_med gate (thr=0.6, shift=1)",
    "rows": rows,
    "best": best,
}
(out_dir / "variants_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

if best is not None:
    # Copy the selected config JSON into OUT_DIR for E0227.
    best_variant = best["variant"]
    cfg_path = out_dir / best_variant / "config.json"
    if cfg_path.exists():
        (out_dir / "best_config.json").write_text(cfg_path.read_text(encoding="utf-8"), encoding="utf-8")
    else:
        # Fallback: copy from the original best_config (if the run didn't write config.json for some reason).
        raise SystemExit(f"missing {cfg_path}")

print("[e0226] wrote", out_dir / "variants_summary.json")
print("[e0226] best", best)
PY

echo "OK: ${OUT_DIR}"
