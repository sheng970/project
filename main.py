"""端到端主入口：线性回归 + 逻辑回归两条 PyTorch 风格流水线。

整体风格严格参考 PyTorch 框架：

  ┌────────────┐    ┌────────────┐    ┌────────────┐    ┌────────────┐
  │  DataSet   │ →  │  DataLoader│ →  │   Model    │ →  │    Loss    │
  │  (X, y)    │    │  切 batch   │    │  nn.Module │    │  nn.Module │
  └────────────┘    └────────────┘    └────────────┘    └────────────┘
                                                              ↓
                                                       ┌────────────┐
                                                       │ Optimizer  │
                                                       │ 7 选 1     │
                                                       └────────────┘

训练循环（大 for + 小 for）::

    for epoch in range(epochs):               # 大 for：epoch
        for x, y in dataloader:                # 小 for：batch
            optimizer.zero_grad()
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            loss.backward()                    # autograd 算 param.grad
            optimizer.step()                   # 自定义 Optimizer 更新参数

两条流水线：
- 线性回归：合成数据 ``y = 2x1 + 3x2 + 1 + ε``，MSELoss
- 逻辑回归：``Social_Network_Ads.xlsx``，BCELoss

7 个梯度下降优化器（详见 ``data/optim/optimizers.py``）：
BGD / SGD / MBGD / Momentum / AdaGrad / RMSProp / Adam

命令行指定示例::

    python main.py                     # 默认 Adam
    python main.py --optimizer MBGD    # 改用 Mini-Batch
    python main.py --task linear       # 只跑线性回归
    python main.py --task logistic     # 只跑逻辑回归
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from data.classify.dataset import SocialNetworkAdsDataset  # noqa: E402
from data.classify.features import train_test_split_indices  # noqa: E402
from data.loss.loss import BCELoss, MSELoss  # noqa: E402
from data.model.model import LinearRegression, LogisticRegression  # noqa: E402
from data.optim.optimizers import Optimizer, build_optimizer  # noqa: E402
from data.regression.dataset import LinearRegressionDataset  # noqa: E402


# =====================================================================
# 训练流程（PyTorch 风格：大 for epoch + 小 for batch）
# =====================================================================


@dataclass
class TrainConfig:
    """统一训练配置：线性/逻辑回归共用。"""

    epochs: int = 200
    batch_size: int = 32
    learning_rate: float = 0.05
    log_every: int = 20
    optimizer_name: str = "Adam"
    device: str = "auto"
    seed: int = 42


def get_device(device: str = "auto") -> torch.device:
    """解析设备字符串。``auto`` 时优先 CUDA。"""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: Optimizer,
    config: TrainConfig,
    metric_fn: Callable[[torch.Tensor, torch.Tensor], float] | None = None,
) -> dict[str, list[float]]:
    """执行完整训练流程（大 for + 小 for）。

    Args:
        model: 任意 ``nn.Module`` 子类。
        train_loader: 训练集 ``DataLoader``。
        test_loader: 测试集 ``DataLoader``。
        loss_fn: 损失函数 ``nn.Module``。
        optimizer: 自定义 ``Optimizer``（7 选 1）。
        config: 训练超参。
        metric_fn: 评估指标（如分类准确率），返回 float。

    Returns:
        训练历史 ``{"loss": [...], "val_loss": [...], "val_metric": [...]}``。
    """
    device = get_device(config.device)
    model.to(device)
    history: dict[str, list[float]] = {"loss": [], "val_loss": [], "val_metric": []}

    for epoch in range(1, config.epochs + 1):                       # 大 for：epoch
        # ---------- 训练：每个 batch 一次 zero_grad → forward → loss → backward → step
        model.train()
        total, n = 0.0, 0
        for x, y in train_loader:                                    # 小 for：batch
            x = x.to(device)
            y = y.to(device).view(-1, 1)                             # 重要：让 y 与 y_pred 同形 (N,1)，避免广播成 (N,N)
            optimizer.zero_grad()
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()
            total += loss.item() * x.size(0)
            n += x.size(0)
        train_loss = total / max(n, 1)

        # ---------- 评估：每个 epoch 末
        model.eval()
        total, n = 0.0, 0
        metric_total = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device)
                y = y.to(device).view(-1, 1)                         # 与训练保持一致
                y_pred = model(x)
                loss = loss_fn(y_pred, y)
                total += loss.item() * x.size(0)
                n += x.size(0)
                if metric_fn is not None:
                    metric_total += metric_fn(y_pred, y) * x.size(0)
        val_loss = total / max(n, 1)
        val_metric = metric_total / max(n, 1) if metric_fn is not None else float("nan")

        history["loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_metric"].append(val_metric)

        if epoch % config.log_every == 0 or epoch == config.epochs:
            print(
                f"epoch {epoch:4d}/{config.epochs} "
                f"loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"val_metric={val_metric:.4f}"
            )
    return history


# =====================================================================
# 流水线 1：线性回归（合成数据）
# =====================================================================


def accuracy_regression(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """回归任务的「相对误差」指标（越接近 1 越好），仅供展示 metric 接口。"""
    denom = y_true.abs().clamp(min=1e-3)
    rel = 1.0 - (y_pred - y_true).abs() / denom
    return rel.mean().item()


def run_linear_regression(config: TrainConfig) -> None:
    """流水线 1：合成数据 → LinearRegression → MSELoss → 自定义 Optimizer。"""
    print("\n" + "=" * 70)
    print("[task 1] 线性回归 — 合成数据 y = 2·x1 + 3·x2 + 1 + ε")
    print("=" * 70)

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # ----- Dataset + DataLoader
    dataset = LinearRegressionDataset(n_samples=200, n_features=2, noise=0.1, seed=config.seed)
    train_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)

    print(
        f"[data] n_samples={len(dataset)} feature_dim={dataset.features.shape[1]} "
        f"true_w={dataset.true_w.squeeze().tolist()} true_b={dataset.true_b}"
    )

    # ----- Model + Loss + Optimizer
    device = get_device(config.device)
    model = LinearRegression(input_dim=2).to(device)
    loss_fn = MSELoss()
    optimizer = build_optimizer(
        config.optimizer_name, model.parameters(), lr=config.learning_rate
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"[model] {model.name} params={n_params} "
        f"loss={loss_fn.__class__.__name__} optimizer={optimizer.name} lr={config.learning_rate}"
    )

    # ----- 训练（大 for + 小 for）
    history = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        config=config,
        metric_fn=accuracy_regression,
    )

    # ----- 验证：学到的参数 vs 真实参数
    learned_w = model.linear.weight.detach().cpu().squeeze().tolist()
    learned_b = model.linear.bias.detach().cpu().item()
    print(
        f"[check] learned_w={[round(x, 3) for x in learned_w]} "
        f"learned_b={learned_b:.3f} "
        f"vs true_w={dataset.true_w.squeeze().tolist()} true_b={dataset.true_b}"
    )
    print(
        f"[final] train_loss={history['loss'][-1]:.4f} "
        f"val_loss={history['val_loss'][-1]:.4f}"
    )


# =====================================================================
# 流水线 2：逻辑回归（Social_Network_Ads）
# =====================================================================


def accuracy_classification(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """二分类准确率：``y_pred >= 0.5`` 与 ``y_true`` 的命中率。"""
    return (y_pred.round() == y_true).float().mean().item()


def run_logistic_regression(config: TrainConfig) -> None:
    """流水线 2：Social_Network_Ads → LogisticRegression → BCELoss → 自定义 Optimizer。"""
    print("\n" + "=" * 70)
    print("[task 2] 逻辑回归 — Social_Network_Ads（年龄 + 预估薪资 → 是否购买）")
    print("=" * 70)

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # ----- Dataset + DataLoader
    data_path = ROOT / "data" / "raw" / "Social_Network_Ads.xlsx"
    dataset = SocialNetworkAdsDataset(data_path)
    train_idx, test_idx = train_test_split_indices(
        n_samples=len(dataset), test_size=0.2, random_state=config.seed
    )
    # 训练集统计量做 z-score 标准化（避免数据泄露）
    x_train = dataset.features[train_idx]
    mean = x_train.mean(dim=0)
    std = x_train.std(dim=0) + 1e-8
    dataset.features = (dataset.features - mean) / std
    print(f"[data] n={len(dataset)} pos={int(dataset.targets.sum().item())} scale={std.tolist()}")

    train_loader = DataLoader(
        torch.utils.data.Subset(dataset, train_idx.tolist()),
        batch_size=config.batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        torch.utils.data.Subset(dataset, test_idx.tolist()),
        batch_size=config.batch_size,
        shuffle=False,
    )

    # ----- Model + Loss + Optimizer
    device = get_device(config.device)
    model = LogisticRegression(input_dim=2).to(device)
    loss_fn = BCELoss()
    optimizer = build_optimizer(
        config.optimizer_name, model.parameters(), lr=config.learning_rate
    )
    print(
        f"[model] {model.name} loss={loss_fn.__class__.__name__} "
        f"optimizer={optimizer.name} lr={config.learning_rate}"
    )

    # ----- 训练
    history = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        config=config,
        metric_fn=accuracy_classification,
    )
    print(
        f"[final] train_loss={history['loss'][-1]:.4f} "
        f"val_loss={history['val_loss'][-1]:.4f} "
        f"val_acc={history['val_metric'][-1]:.4f}"
    )

    # ----- 落盘
    weights_path = ROOT / "data" / "runs" / f"weights_{config.optimizer_name}.pt"
    model.save(weights_path)
    print(f"[save] weights → {weights_path}")


# =====================================================================
# 入口
# =====================================================================


def parse_args() -> tuple[TrainConfig, str]:
    """解析命令行。"""
    parser = argparse.ArgumentParser(description="PyTorch 风格双流水线训练（线性+逻辑）")
    parser.add_argument(
        "--task",
        default="all",
        choices=["all", "linear", "logistic"],
        help="跑哪条流水线（默认 all）",
    )
    parser.add_argument(
        "--optimizer",
        default="Adam",
        choices=["BGD", "SGD", "MBGD", "Momentum", "AdaGrad", "RMSProp", "Adam"],
        help="从 7 个自实现梯度下降优化器中选 1 个（默认 Adam）",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(
        f"[config] task={args.task} optimizer={args.optimizer} "
        f"epochs={args.epochs} batch_size={args.batch_size} lr={args.lr}"
    )
    config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        log_every=args.log_every,
        optimizer_name=args.optimizer,
        seed=args.seed,
    )
    return config, args.task


def main() -> None:
    """总入口：解析参数 → 跑指定流水线。"""
    config, task = parse_args()

    if task in ("all", "linear"):
        run_linear_regression(config)
    if task in ("all", "logistic"):
        run_logistic_regression(config)


if __name__ == "__main__":
    main()
