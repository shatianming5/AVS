from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 3
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 0.0


@torch.no_grad()
def segment_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    correct = (preds == targets).float().mean().item()
    return float(correct)


def train_per_segment_classifier(
    *,
    model: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    cfg: TrainConfig,
) -> dict:
    device = next(model.parameters()).device
    model.train()

    # Ensure data is on the same device as the model.
    x_train = x_train.to(device)
    y_train = y_train.to(device)
    x_val = x_val.to(device)
    y_val = y_val.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    n = x_train.shape[0]
    steps_per_epoch = max(1, (n + cfg.batch_size - 1) // cfg.batch_size)

    for _epoch in range(cfg.epochs):
        perm = torch.randperm(n, device=device)
        for i in range(steps_per_epoch):
            idx = perm[i * cfg.batch_size : (i + 1) * cfg.batch_size]
            xb = x_train[idx]
            yb = y_train[idx]

            logits = model(xb)
            loss = loss_fn(logits.view(-1, logits.shape[-1]), yb.view(-1))

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        val_logits = model(x_val)
        val_acc = segment_accuracy(val_logits, y_val)

        preds = val_logits.argmax(dim=-1)
        correct = (preds == y_val)
        val_acc_by_sample = correct.float().mean(dim=1).detach().cpu().numpy().astype("float32")

        event_mask = y_val != 0
        num_event = event_mask.sum().item()
        if int(num_event) > 0:
            event_correct = (correct & event_mask).sum().item()
            val_acc_event = float(event_correct / float(num_event))
        else:
            val_acc_event = None

        # Per-sample event-only accuracy. Use None for samples without any event segments.
        event_counts = event_mask.sum(dim=1)
        event_correct_counts = (correct & event_mask).sum(dim=1)
        ev = torch.empty_like(event_correct_counts, dtype=torch.float32)
        ev[:] = float("nan")
        has_event = event_counts > 0
        ev[has_event] = event_correct_counts[has_event].float() / event_counts[has_event].float()
        val_acc_event_by_sample = ev.detach().cpu().numpy().astype("float32")

    return {
        "val_acc": float(val_acc),
        "val_acc_event": val_acc_event,
        "val_acc_by_sample": [float(x) for x in val_acc_by_sample.tolist()],
        "val_acc_event_by_sample": [None if float(x) != float(x) else float(x) for x in val_acc_event_by_sample.tolist()],
    }
