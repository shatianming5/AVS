from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from avs.datasets.avqa import AVQAIndex
from avs.datasets.intentqa import load_intentqa_split


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                out.append(json.loads(s))
            except Exception as e:  # noqa: BLE001
                raise ValueError(f"failed to parse jsonl at {path}:{ln}: {e}") from e
    return out


def _parse_eval_labels(task: str, rows: list[dict[str, Any]]) -> tuple[list[int], int]:
    labels_by_item: dict[str, int] = {}
    if task == "intentqa":
        for r in rows:
            key = f"{r.get('video_id')}|{r.get('qid')}"
            if key in labels_by_item:
                continue
            v = r.get("answer_idx")
            if v is None:
                continue
            labels_by_item[key] = int(v)
    elif task == "avqa":
        for r in rows:
            key = str(r.get("item_id"))
            if key in labels_by_item:
                continue
            v = r.get("label_idx")
            if v is None:
                continue
            labels_by_item[key] = int(v)
    elif task == "egoschema":
        for r in rows:
            key = f"{r.get('video_idx')}|{r.get('question_idx')}"
            if key in labels_by_item:
                continue
            v = r.get("answer_idx")
            if v is None:
                continue
            labels_by_item[key] = int(v)
    else:
        raise ValueError(f"unsupported task: {task}")
    labels = list(labels_by_item.values())
    if not labels:
        raise ValueError(f"no labels found in predictions rows for task={task}")
    n_classes = int(max(labels) + 1)
    return labels, n_classes


def _intentqa_train_prior() -> Counter[int]:
    cnt: Counter[int] = Counter()
    for it in load_intentqa_split("train"):
        cnt[int(it.answer_idx)] += 1
    return cnt


def _avqa_train_prior(meta_dir: Path) -> Counter[int]:
    idx = AVQAIndex.from_meta_dir(meta_dir)
    cnt: Counter[int] = Counter()
    for it in idx.train:
        cnt[int(it.answer)] += 1
    return cnt


def _normalized_probs(cnt: Counter[int], n_classes: int) -> list[float]:
    tot = float(sum(cnt.values()))
    if tot <= 0:
        return [0.0] * int(n_classes)
    return [float(cnt.get(i, 0) / tot) for i in range(int(n_classes))]


def _safe_acc(pred: int, labels: list[int]) -> float:
    if not labels:
        return 0.0
    hit = sum(1 for y in labels if int(y) == int(pred))
    return float(hit / len(labels))


def _parse_metrics_summary(metrics_path: Path | None) -> dict[str, float]:
    if metrics_path is None:
        return {}
    obj = json.loads(metrics_path.read_text(encoding="utf-8"))
    sm = obj.get("summary")
    out: dict[str, float] = {}
    if isinstance(sm, list):
        for row in sm:
            if not isinstance(row, dict):
                continue
            m = str(row.get("method", ""))
            a = row.get("acc")
            if m and a is not None:
                out[m] = float(a)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="Compute answer-prior baseline on evaluated QA subset.")
    p.add_argument("--task", type=str, required=True, choices=["intentqa", "avqa", "egoschema"])
    p.add_argument("--predictions", type=Path, required=True, help="Path to predictions.jsonl")
    p.add_argument("--metrics", type=Path, default=None, help="Optional metrics.json to compare with existing methods.")
    p.add_argument("--avqa-meta-dir", type=Path, default=Path("data/AVQA/meta"))
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()

    rows = _read_jsonl(Path(args.predictions))
    labels_eval, n_classes = _parse_eval_labels(str(args.task), rows)
    eval_cnt: Counter[int] = Counter(int(y) for y in labels_eval)
    eval_probs = _normalized_probs(eval_cnt, n_classes)

    if str(args.task) == "intentqa":
        prior_cnt = _intentqa_train_prior()
        prior_source = "train_split"
    elif str(args.task) == "avqa":
        prior_cnt = _avqa_train_prior(Path(args.avqa_meta_dir))
        prior_source = "train_split"
    else:
        prior_cnt = Counter(eval_cnt)
        prior_source = "eval_split_proxy_no_train_labels"

    prior_probs = _normalized_probs(prior_cnt, n_classes)
    top_answer = int(max(range(n_classes), key=lambda i: prior_probs[i]))
    top_prior_acc = _safe_acc(top_answer, labels_eval)

    expected_sampling_acc = 0.0
    for i in range(n_classes):
        expected_sampling_acc += float(prior_probs[i] * eval_probs[i])

    methods_acc = _parse_metrics_summary(args.metrics)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "ok": True,
        "task": str(args.task),
        "predictions_jsonl": str(args.predictions),
        "metrics_json": str(args.metrics) if args.metrics else None,
        "n_eval": int(len(labels_eval)),
        "n_classes": int(n_classes),
        "prior_source": prior_source,
        "prior_counts": {str(k): int(v) for k, v in sorted(prior_cnt.items())},
        "eval_counts": {str(k): int(v) for k, v in sorted(eval_cnt.items())},
        "prior_probs": [float(x) for x in prior_probs],
        "eval_probs": [float(x) for x in eval_probs],
        "top_answer_idx": int(top_answer),
        "answer_prior_acc": float(top_prior_acc),
        "expected_sampling_acc": float(expected_sampling_acc),
        "methods_acc": methods_acc,
        "delta_vs_answer_prior": {m: float(a - top_prior_acc) for m, a in methods_acc.items()},
        "artifacts": {
            "bias_baselines_json": str(out_dir / "bias_baselines.json"),
            "bias_baselines_md": str(out_dir / "bias_baselines.md"),
        },
    }

    md_lines = [
        "# QA Bias Baselines",
        "",
        f"- task: `{str(args.task)}`",
        f"- n_eval: `{len(labels_eval)}`",
        f"- prior_source: `{prior_source}`",
        "",
        "## Answer Prior",
        "",
        f"- top_answer_idx: `{top_answer}`",
        f"- answer_prior_acc: `{top_prior_acc:.4f}`",
        f"- expected_sampling_acc: `{expected_sampling_acc:.4f}`",
        "",
        "## Compared Methods",
        "",
        "| method | acc | delta_vs_answer_prior |",
        "|---|---:|---:|",
    ]
    for m in sorted(methods_acc.keys()):
        a = float(methods_acc[m])
        d = float(a - top_prior_acc)
        md_lines.append(f"| `{m}` | {a:.4f} | {d:+.4f} |")
    md_lines.append("")

    (out_dir / "bias_baselines.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "bias_baselines.md").write_text("\n".join(md_lines), encoding="utf-8")
    print(out_dir / "bias_baselines.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
