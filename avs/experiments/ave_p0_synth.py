from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from avs.models.per_segment_mlp import PerSegmentMLP
from avs.sampling.plans import equal_token_budget_anchored_plan
from avs.train.synthetic_ave import SyntheticAVEConfig, make_synthetic_ave
from avs.train.train_loop import TrainConfig, train_per_segment_classifier


@dataclass(frozen=True)
class P0SynthResult:
    seed: int
    uniform_acc: float
    random_acc: float
    anchored_acc: float


def _run_one(seed: int, *, device: torch.device, cfg: TrainConfig) -> P0SynthResult:
    synth = SyntheticAVEConfig(num_samples=256, num_segments=10, num_classes=8, feat_dim=32, seed=seed)

    plan_uniform = equal_token_budget_anchored_plan(num_segments=10, anchors=[])
    plan_random = equal_token_budget_anchored_plan(num_segments=10, anchors=[0, 1])
    plan_anchored = equal_token_budget_anchored_plan(num_segments=10, anchors=[6, 7])

    def _train_eval(plan) -> float:
        x, y = make_synthetic_ave(synth, plan=plan, device=device)
        split = int(0.8 * synth.num_samples)
        x_train, y_train = x[:split], y[:split]
        x_val, y_val = x[split:], y[split:]
        model = PerSegmentMLP(in_dim=synth.feat_dim, num_classes=synth.num_classes, hidden_dim=128).to(device)
        m = train_per_segment_classifier(
            model=model,
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            cfg=cfg,
        )
        return float(m["val_acc"])

    return P0SynthResult(
        seed=seed,
        uniform_acc=_train_eval(plan_uniform),
        random_acc=_train_eval(plan_random),
        anchored_acc=_train_eval(plan_anchored),
    )


def _try_paired_ttest(a: np.ndarray, b: np.ndarray) -> dict | None:
    if a.shape[0] < 2 or b.shape[0] < 2:
        return None
    try:
        from scipy.stats import ttest_rel

        r = ttest_rel(a, b)
        return {"t": float(r.statistic), "p": float(r.pvalue)}
    except Exception:
        return None


def run_p0_synth(*, seeds: list[int], out_dir: Path, device: str = "cpu") -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = TrainConfig(epochs=12, batch_size=32, lr=2e-3)

    dev = torch.device(device)
    rows = [_run_one(s, device=dev, cfg=cfg) for s in seeds]

    uniform = np.array([r.uniform_acc for r in rows], dtype=np.float64)
    random = np.array([r.random_acc for r in rows], dtype=np.float64)
    anchored = np.array([r.anchored_acc for r in rows], dtype=np.float64)

    summary = {
        "uniform": {"mean": float(uniform.mean()), "std": float(uniform.std(ddof=1)) if len(seeds) > 1 else 0.0},
        "random_top2": {"mean": float(random.mean()), "std": float(random.std(ddof=1)) if len(seeds) > 1 else 0.0},
        "anchored_top2": {"mean": float(anchored.mean()), "std": float(anchored.std(ddof=1)) if len(seeds) > 1 else 0.0},
        "paired_ttest": {
            "anchored_vs_uniform": _try_paired_ttest(anchored, uniform),
            "anchored_vs_random": _try_paired_ttest(anchored, random),
        },
    }

    payload = {
        "seeds": seeds,
        "plans": {
            "uniform": equal_token_budget_anchored_plan(num_segments=10, anchors=[]).to_jsonable(),
            "random_top2": equal_token_budget_anchored_plan(num_segments=10, anchors=[0, 1]).to_jsonable(),
            "anchored_top2": equal_token_budget_anchored_plan(num_segments=10, anchors=[6, 7]).to_jsonable(),
        },
        "rows": [r.__dict__ for r in rows],
        "summary": summary,
    }
    out_path = out_dir / "metrics.json"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return out_path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="AVE-P0 synthetic experiment (equal token budget baselines).")
    p.add_argument("--out-dir", type=Path, default=Path("runs") / f"AVE_P0_synth_{time.strftime('%Y%m%d-%H%M%S')}")
    p.add_argument("--seeds", type=str, default="0,1,2")
    p.add_argument("--device", type=str, default="cpu")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    out_path = run_p0_synth(seeds=seeds, out_dir=args.out_dir, device=args.device)
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
