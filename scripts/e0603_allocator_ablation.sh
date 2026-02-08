#!/usr/bin/env bash
set -euo pipefail

# E0603: Stage-2 solver ablation (greedy vs Lagrangian knapsack) on synthetic windows.
#
# Outputs:
#   runs/E0603_allocator_ablation_<ts>/allocator_ablation.json

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

OUT_DIR="${OUT_DIR:-runs/E0603_allocator_ablation_$(date +%Y%m%d-%H%M%S)}"
SEED="${SEED:-0}"
NUM_WINDOWS="${NUM_WINDOWS:-12}"
BUDGET="${BUDGET:-20000}"

mkdir -p "${OUT_DIR}"

python -m avs.experiments.allocator_solver_ablation \
  --out-dir "${OUT_DIR}" \
  --seed "${SEED}" \
  --num-windows "${NUM_WINDOWS}" \
  --budget "${BUDGET}"

