"""训练/评估流程（PyTorch 风格：``zero_grad → forward → loss → backward → step``）。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader


@dataclass
class TrainingConfig:
    """训练超参数。"""

    learning_rate: float = 0.05
    epochs: int = 100
    batch_size: int = 32
    device: str = "auto"
    log_every: int = 20
    loss_curve: list[float] = field(default_factory=list)


def get_device(device: str = "auto") -> torch.device:
    """解析设备字符串；``auto`` 时优先选择 CUDA。"""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """训练一个 epoch，返回平均损失。"""
    model.train()
    total_loss = 0.0
    n = 0
    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device).unsqueeze(1)
        optimizer.zero_grad()
        y_pred = model(x_batch)
        loss = loss_fn(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x_batch.size(0)
        n += x_batch.size(0)
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """评估：返回 ``(平均损失, 准确率)``。"""
    model.eval()
    total_loss = 0.0
    n = 0
    correct = 0
    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device).unsqueeze(1)
        y_pred = model(x_batch)
        loss = loss_fn(y_pred, y_batch)
        total_loss += loss.item() * x_batch.size(0)
        n += x_batch.size(0)
        correct += (y_pred.round() == y_batch).float().sum().item()
    return total_loss / max(n, 1), correct / max(n, 1)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: TrainingConfig = TrainingConfig(),
) -> dict[str, list[float]]:
    """执行完整训练流程并返回训练历史。"""
    device = get_device(config.device)
    model.to(device)
    history: dict[str, list[float]] = {"loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, config.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)
        history["loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        if epoch % config.log_every == 0 or epoch == config.epochs:
            print(
                f"epoch {epoch:4d}/{config.epochs} "
                f"loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            )
    return history
