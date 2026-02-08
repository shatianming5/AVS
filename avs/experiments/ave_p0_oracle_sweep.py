from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

from avs.datasets.ave import AVEIndex, ensure_ave_meta
from avs.datasets.layout import ave_paths
from avs.experiments.ave_p0 import P0Config, run_p0_from_caches
from avs.train.train_loop import TrainConfig


def _read_ids_file(path: Path, limit: int | None) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        s = str(line).strip()
        if not s:
            continue
        if s in seen:
            continue
        seen.add(s)
        ids.append(s)
        if limit is not None and len(ids) >= int(limit):
            break
    return ids


def _split_ids(index: AVEIndex, split: str, limit: int | None) -> list[str]:
    ids = index.splits[str(split)]
    if limit is not None:
        ids = ids[: int(limit)]
    return [index.clips[int(i)].video_id for i in ids]


@dataclass(frozen=True)
class OracleSweepConfig:
    name: str
    low_res: int
    base_res: int
    high_res: int
    head: str
    head_hidden_dim: int = 128
    head_dropout: float = 0.0
    temporal_kernel_size: int = 3
    max_high_anchors: int | None = None

    def to_jsonable(self) -> dict:
        return {
            "name": str(self.name),
            "low_res": int(self.low_res),
            "base_res": int(self.base_res),
            "high_res": int(self.high_res),
            "head": str(self.head),
            "head_hidden_dim": int(self.head_hidden_dim),
            "head_dropout": float(self.head_dropout),
            "temporal_kernel_size": int(self.temporal_kernel_size),
            "max_high_anchors": int(self.max_high_anchors) if self.max_high_anchors is not None else None,
        }


def _default_configs() -> list[OracleSweepConfig]:
    # Keep this list small and decision-complete: no hidden auto-search.
    return [
        OracleSweepConfig(name="112_224_448_mlp", low_res=112, base_res=224, high_res=448, head="mlp"),
        OracleSweepConfig(name="112_224_448_temporal_k3", low_res=112, base_res=224, high_res=448, head="temporal_conv", temporal_kernel_size=3),
        OracleSweepConfig(name="112_224_448_temporal_k3_max1", low_res=112, base_res=224, high_res=448, head="temporal_conv", temporal_kernel_size=3, max_high_anchors=1),
        OracleSweepConfig(name="160_224_352_mlp", low_res=160, base_res=224, high_res=352, head="mlp"),
        OracleSweepConfig(name="160_224_352_temporal_k3", low_res=160, base_res=224, high_res=352, head="temporal_conv", temporal_kernel_size=3),
        OracleSweepConfig(name="160_224_352_temporal_k5", low_res=160, base_res=224, high_res=352, head="temporal_conv", temporal_kernel_size=5),
    ]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Oracle ceiling sweep for AVE-P0 (uniform vs oracle_top2 under equal token budget).")
    p.add_argument("--meta-dir", type=Path, default=ave_paths().meta_dir)
    p.add_argument("--caches-dir", type=Path, required=True, help="Dir containing <clip_id>.npz feature caches")

    p.add_argument("--split-train", type=str, default="train", choices=["train", "val", "test"])
    p.add_argument("--split-eval", type=str, default="val", choices=["train", "val", "test"])
    p.add_argument("--limit-train", type=int, default=None)
    p.add_argument("--limit-eval", type=int, default=None)
    p.add_argument("--train-ids-file", type=Path, default=None, help="Optional file with one train video_id per line.")
    p.add_argument("--eval-ids-file", type=Path, default=None, help="Optional file with one eval video_id per line.")
    p.add_argument("--allow-missing", action="store_true", help="Skip clips with missing caches instead of failing.")

    p.add_argument("--seeds", type=str, default="0,1,2,3,4,5,6,7,8,9")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--train-device", type=str, default="cuda:0")

    p.add_argument("--out-dir", type=Path, default=Path("runs") / f"E0010_oracle_ceiling_{time.strftime('%Y%m%d-%H%M%S')}")
    p.add_argument("--configs-json", type=Path, default=None, help="Optional JSON list of configs to sweep (overrides defaults).")
    return p


def _load_configs(path: Path) -> list[OracleSweepConfig]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, list):
        raise ValueError("configs-json must be a JSON list")
    out: list[OracleSweepConfig] = []
    for item in obj:
        if not isinstance(item, dict):
            raise ValueError("configs-json list items must be objects")
        out.append(
            OracleSweepConfig(
                name=str(item["name"]),
                low_res=int(item["low_res"]),
                base_res=int(item["base_res"]),
                high_res=int(item["high_res"]),
                head=str(item.get("head", "mlp")),
                head_hidden_dim=int(item.get("head_hidden_dim", 128)),
                head_dropout=float(item.get("head_dropout", 0.0)),
                temporal_kernel_size=int(item.get("temporal_kernel_size", 3)),
                max_high_anchors=int(item["max_high_anchors"]) if item.get("max_high_anchors") is not None else None,
            )
        )
    if not out:
        raise ValueError("configs-json contained no configs")
    return out


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    ensure_ave_meta(args.meta_dir)
    index = AVEIndex.from_meta_dir(args.meta_dir)

    if args.train_ids_file is not None:
        train_ids = _read_ids_file(args.train_ids_file, args.limit_train)
    else:
        train_ids = _split_ids(index, args.split_train, args.limit_train)

    if args.eval_ids_file is not None:
        eval_ids = _read_ids_file(args.eval_ids_file, args.limit_eval)
    else:
        eval_ids = _split_ids(index, args.split_eval, args.limit_eval)

    cache_dir = Path(args.caches_dir)
    if args.allow_missing:
        cached = {p.stem for p in cache_dir.glob("*.npz")}
        train_ids = [cid for cid in train_ids if cid in cached]
        eval_ids = [cid for cid in eval_ids if cid in cached]

    if not train_ids or not eval_ids:
        raise SystemExit(f"no usable ids after filtering (train={len(train_ids)} eval={len(eval_ids)})")

    clip_by_id = {c.video_id: c for c in index.clips}
    labels_by_clip: dict[str, list[int]] = {}
    for cid in sorted(set(train_ids + eval_ids)):
        clip = clip_by_id.get(cid)
        if clip is None:
            continue
        labels_by_clip[cid] = [int(x) for x in index.segment_labels(clip)]

    seeds = [int(s) for s in str(args.seeds).split(",") if str(s).strip()]
    if len(seeds) < 2:
        raise SystemExit("--seeds must contain at least 2 seeds to compute paired p-values")

    train_cfg = TrainConfig(epochs=int(args.epochs), batch_size=int(args.batch_size), lr=float(args.lr), weight_decay=float(args.weight_decay))

    configs = _load_configs(args.configs_json) if args.configs_json is not None else _default_configs()

    rows: list[dict] = []
    for cfg in configs:
        run_dir = args.out_dir / cfg.name
        run_dir.mkdir(parents=True, exist_ok=True)

        metrics = run_p0_from_caches(
            clip_ids_train=train_ids,
            clip_ids_eval=eval_ids,
            labels_by_clip=labels_by_clip,
            caches_dir=cache_dir,
            audio_dir=None,
            cfg=P0Config(
                k=2,
                low_res=int(cfg.low_res),
                base_res=int(cfg.base_res),
                high_res=int(cfg.high_res),
                patch_size=16,
                max_high_anchors=cfg.max_high_anchors,
                head=str(cfg.head),
                head_hidden_dim=int(cfg.head_hidden_dim),
                head_dropout=float(cfg.head_dropout),
                temporal_kernel_size=int(cfg.temporal_kernel_size),
            ),
            baselines=["uniform", "oracle_top2"],
            seeds=seeds,
            train_cfg=train_cfg,
            train_device=str(args.train_device),
            num_classes=index.num_classes,
            num_segments=10,
        )

        metrics_path = run_dir / "metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        summ = metrics.get("summary") or {}
        ptt = metrics.get("paired_ttest") or {}
        row = {
            "config": cfg.to_jsonable(),
            "metrics_path": str(metrics_path),
            "uniform": summ.get("uniform"),
            "oracle_top2": summ.get("oracle_top2"),
            "paired_ttest": {"oracle_vs_uniform": ptt.get("oracle_vs_uniform")},
        }
        try:
            u = float(summ.get("uniform", {}).get("mean"))
            o = float(summ.get("oracle_top2", {}).get("mean"))
            row["oracle_minus_uniform_mean"] = float(o - u)
        except Exception:
            row["oracle_minus_uniform_mean"] = None
        rows.append(row)

    out = {
        "ok": True,
        "meta_dir": str(args.meta_dir),
        "caches_dir": str(cache_dir),
        "split_train": str(args.split_train),
        "split_eval": str(args.split_eval),
        "num_train_ids": int(len(train_ids)),
        "num_eval_ids": int(len(eval_ids)),
        "seeds": seeds,
        "train_cfg": {"epochs": int(train_cfg.epochs), "batch_size": int(train_cfg.batch_size), "lr": float(train_cfg.lr), "weight_decay": float(train_cfg.weight_decay)},
        "results": rows,
    }
    out_path = args.out_dir / "oracle_ceiling.json"
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
