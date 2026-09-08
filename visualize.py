"""把训练结果可视化为图片（matplotlib, Agg 后端 → PNG）。

生成 3 张图到 ``data/figures/``：

1. ``optimizer_curves.png``    7 个梯度下降优化器在同一逻辑回归任务上的损失下降对比
2. ``linear_regression.png``   线性回归：损失曲线 + 拟合散点 + 学到的参数 vs 真实参数
3. ``logistic_boundary.png``   逻辑回归：决策边界（概率等高线）+ 原始数据点

运行::

    python visualize.py                 # 全部 3 张
    python visualize.py --only curves   # 只画损失曲线对比
    python visualize.py --only linear
    python visualize.py --only logistic
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 无 GUI 后端，直接输出 PNG

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from data.classify.dataset import SocialNetworkAdsDataset  # noqa: E402
from data.classify.features import train_test_split_indices  # noqa: E402
from data.loss.loss import BCELoss, MSELoss  # noqa: E402
from data.model.model import LinearRegression, LogisticRegression  # noqa: E402
from data.optim.optimizers import (  # noqa: E402
    BGD,
    SGD,
    MBGD,
    Momentum,
    AdaGrad,
    RMSProp,
    Adam,
    Optimizer,
)
from data.regression.dataset import LinearRegressionDataset  # noqa: E402

FIG_DIR = ROOT / "data" / "figures"
DATA_PATH = ROOT / "data" / "raw" / "Social_Network_Ads.xlsx"
SEED = 42
TEST_SIZE = 0.2
ALL_OPTIMIZERS: list[tuple[str, type[Optimizer], float]] = [
    ("BGD", BGD, 0.05),
    ("SGD", SGD, 0.01),
    ("MBGD", MBGD, 0.05),
    ("Momentum", Momentum, 0.05),
    ("AdaGrad", AdaGrad, 0.1),
    ("RMSProp", RMSProp, 0.01),
    ("Adam", Adam, 0.05),
]

# 图内文字统一英文，避免中文字体缺字形（Windows 无 display 环境下最稳妥）
plt.rcParams.update(
    {
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)
PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#17becf", "#e377c2",
]


# =====================================================================
# 工具
# =====================================================================


def seed_all(seed: int = SEED) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def ensure_fig_dir() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def fit_history(
    model: nn.Module,
    dataset: Dataset,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    loss_fn: nn.Module,
    optimizer_cls: type[Optimizer],
    lr: float,
    batch_size: int,
    epochs: int,
) -> tuple[dict[str, list[float]], float]:
    """训练并返回 ``{"loss": [...], "val_loss": [...]}`` 与最终 val_acc。"""
    train_loader = DataLoader(Subset(dataset, train_idx.tolist()), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(Subset(dataset, test_idx.tolist()), batch_size=batch_size, shuffle=False)

    optimizer = optimizer_cls(model.parameters(), lr=lr)
    history: dict[str, list[float]] = {"loss": [], "val_loss": []}
    best_acc = 0.0

    for _ in range(epochs):
        model.train()
        losses, n = 0.0, 0
        for x, y in train_loader:
            y = y.view(-1, 1)
            optimizer.zero_grad()
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()
            losses += loss.item() * x.size(0)
            n += x.size(0)
        history["loss"].append(losses / max(n, 1))

        model.eval()
        vloss, correct, n = 0.0, 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                y = y.view(-1, 1)
                y_pred = model(x)
                vloss += loss_fn(y_pred, y).item() * x.size(0)
                correct += int((y_pred.round() == y).sum().item())
                n += x.size(0)
        history["val_loss"].append(vloss / max(n, 1))
        best_acc = correct / max(n, 1)
    return history, best_acc


def save_fig(fig: plt.Figure, name: str) -> Path:
    ensure_fig_dir()
    out = FIG_DIR / name
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] {out}")
    return out


# =====================================================================
# 图 1：7 个优化器在逻辑回归上的损失曲线对比
# =====================================================================


def plot_optimizer_curves(epochs: int = 150, batch_size: int = 32) -> Path:
    """同一逻辑回归任务，7 个优化器各自重训，画 validation BCE 下降曲线。"""
    dataset = SocialNetworkAdsDataset(DATA_PATH)
    train_idx, test_idx = train_test_split_indices(len(dataset), TEST_SIZE, SEED)

    # 复用 main.py 相同的 z-score 标准化（训练集统计量）
    x_train = dataset.features[train_idx]
    mean = x_train.mean(dim=0)
    std = x_train.std(dim=0) + 1e-8
    dataset.features = (dataset.features - mean) / std

    print(f"[optimizer_curves] epochs={epochs} on logistic regression (BCE)")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for i, (name, cls, lr) in enumerate(ALL_OPTIMIZERS):
        seed_all(SEED)
        model = LogisticRegression(input_dim=2)
        history, acc = fit_history(
            model, dataset, train_idx, test_idx, BCELoss(),
            cls, lr, batch_size, epochs,
        )
        # 曲线用 train loss（val loss 偏噪声且 7 条更容易挤在一起）
        ax.plot(
            np.arange(1, epochs + 1), history["loss"],
            color=PALETTE[i % len(PALETTE)], lw=1.6, label=f"{name} (lr={lr}, acc={acc:.3f})",
        )
        print(f"  {name:9s} lr={lr:<5} final train_loss={history['loss'][-1]:.4f} val_acc={acc:.4f}")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss (Binary Cross Entropy)")
    ax.set_title("Loss Convergence — 7 Gradient Descent Optimizers (Logistic Regression)")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8, ncol=2)
    return save_fig(fig, "optimizer_curves.png")


# =====================================================================
# 图 2：线性回归整体可视化
# =====================================================================


def plot_linear_regression(epochs: int = 150, batch_size: int = 32) -> Path:
    """线性回归三连：损失曲线 / 预测 vs 真值 / 学到参数 vs 真实参数。"""
    dataset = LinearRegressionDataset(n_samples=200, n_features=2, noise=0.1, seed=SEED)
    train_idx, test_idx = train_test_split_indices(len(dataset), TEST_SIZE, SEED)
    # 无重复采样：回归数据不需要 val；此处用全部数据训练、同批评估，只作图示
    train_idx = np.arange(len(dataset))
    test_idx = np.arange(len(dataset))

    seed_all(SEED)
    model = LinearRegression(input_dim=2)
    optimizer = Adam(model.parameters(), lr=0.05)
    history, _ = fit_history(model, dataset, train_idx, test_idx, MSELoss(), Adam, 0.05, batch_size, epochs)

    # ----- 3 个子图
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # (a) 损失曲线
    ax = axes[0]
    ax.plot(np.arange(1, epochs + 1), history["loss"], color=PALETTE[0], lw=1.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("(a) Loss Convergence (Adam)")
    ax.set_ylim(bottom=0)

    # (b) 预测 vs 真值（用未训练后的全量预测）
    with torch.no_grad():
        y_pred = model(dataset.features).squeeze().numpy()
    y_true = dataset.targets.squeeze().numpy()
    ax = axes[1]
    lim = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.scatter(y_true, y_pred, s=14, alpha=0.6, color=PALETTE[0], label="samples")
    ax.plot(lim, lim, ls="--", color="black", lw=1.2, label="y = x")
    r2 = 1.0 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_true.mean()) ** 2)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("True y")
    ax.set_ylabel("Predicted y")
    ax.set_title(f"(b) Fit Quality  R² = {r2:.4f}")
    ax.legend(fontsize=8)

    # (c) 学到参数 vs 真实参数
    ax = axes[2]
    w_true = [float(dataset.true_w[0, 0]), float(dataset.true_w[1, 0]), float(dataset.true_b)]
    w_learned = [
        float(model.linear.weight.data[0, 0]),
        float(model.linear.weight.data[0, 1]),
        float(model.linear.bias.data[0]),
    ]
    labels = ["w₁", "w₂", "b"]
    xpos = np.arange(3)
    w = 0.36
    ax.bar(xpos - w / 2, w_true, width=w, color=PALETTE[1], label="Ground truth")
    ax.bar(xpos + w / 2, w_learned, width=w, color=PALETTE[0], alpha=0.85, label="Learned (Adam)")
    for xi, (t, l) in enumerate(zip(w_true, w_learned)):
        ax.text(xi - w / 2, t, f"{t:.2f}", ha="center", va="bottom", fontsize=8)
        ax.text(xi + w / 2, l, f"{l:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels)
    ax.set_title("(c) Learned vs True Parameters")
    ax.legend(fontsize=8)

    fig.suptitle("Linear Regression on Synthetic Data  y = 2·x₁ + 3·x₂ + 1 + ε", fontsize=12)
    fig.tight_layout()
    return save_fig(fig, "linear_regression.png")


# =====================================================================
# 图 3：逻辑回归决策边界 + 原始数据
# =====================================================================


def plot_logistic_boundary(epochs: int = 200, batch_size: int = 32) -> Path:
    """训练逻辑回归（Adam），在原始 Age / EstimatedSalary 空间画决策区域。"""
    dataset = SocialNetworkAdsDataset(DATA_PATH)
    train_idx, test_idx = train_test_split_indices(len(dataset), TEST_SIZE, SEED)

    # 保留原始坐标用于绘图
    raw_x = dataset.features.numpy().copy()
    raw_y = dataset.targets.numpy().astype(int)

    x_train = dataset.features[train_idx]
    mean = x_train.mean(dim=0).numpy()
    std = x_train.std(dim=0).numpy() + 1e-8
    dataset.features = (dataset.features - mean) / std  # 标准化后训练

    seed_all(SEED)
    model = LogisticRegression(input_dim=2)
    history, acc = fit_history(
        model, dataset, train_idx, test_idx, BCELoss(), Adam, 0.05, batch_size, epochs,
    )

    # 生成原始坐标网格 → 标准化 → 预测概率
    age_lo, age_hi = float(raw_x[:, 0].min()) - 2, float(raw_x[:, 0].max()) + 2
    sal_lo, sal_hi = float(raw_x[:, 1].min()) - 2000, float(raw_x[:, 1].max()) + 2000
    age_grid = np.linspace(age_lo, age_hi, 240)
    sal_grid = np.linspace(sal_lo, sal_hi, 240)
    A, S = np.meshgrid(age_grid, sal_grid)  # A: age 方向
    grid = np.stack([A.ravel(), S.ravel()], axis=1).astype(np.float32)
    grid_z = (grid - mean) / std
    with torch.no_grad():
        prob = model(torch.from_numpy(grid_z)).numpy().reshape(A.shape)

    fig, ax = plt.subplots(figsize=(8.5, 6))
    cmap = plt.get_cmap("RdYlBu_r", 2)
    cf = ax.contourf(A, S, prob, levels=[0.0, 0.5, 1.0], cmap=cmap, alpha=0.35)
    cs = ax.contour(A, S, prob, levels=[0.5], colors="black", linewidths=1.8)
    ax.clabel(cs, fmt="p = 0.5", fontsize=9)

    # 数据点：训练 ○ 测试 ×，颜色按真实标签
    for label, color in [(0, "#2166ac"), (1, "#b2182b")]:
        m = raw_y == label
        ax.scatter(raw_x[m, 0], raw_x[m, 1], c=color, marker="o", s=22,
                   alpha=0.65, edgecolors="white", linewidths=0.4,
                   label=f"class {label} (train+test)")
    ax.set_xlabel("Age")
    ax.set_ylabel("Estimated Salary")
    ax.set_title(
        f"Logistic Regression Decision Boundary (Adam, {epochs} epochs)\n"
        f"test accuracy = {acc:.3f} | threshold p = 0.5"
    )
    ax.legend(fontsize=9, loc="upper left")
    fig.colorbar(cf, ax=ax, ticks=[0.0, 0.5, 1.0]).set_label("P(purchase)")
    fig.tight_layout()
    return save_fig(fig, "logistic_boundary.png")


# =====================================================================
# 入口
# =====================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="把训练结果可视化为 PNG")
    parser.add_argument(
        "--only",
        choices=["curves", "linear", "logistic"],
        default=None,
        help="只生成某一张图（默认全部）",
    )
    parser.add_argument("--epochs", type=int, default=150, help="每条曲线的训练 epoch")
    args = parser.parse_args()

    if args.only is None or args.only == "curves":
        plot_optimizer_curves(epochs=args.epochs)
    if args.only is None or args.only == "linear":
        plot_linear_regression(epochs=args.epochs)
    if args.only is None or args.only == "logistic":
        plot_logistic_boundary(epochs=args.epochs)
    print(f"\nAll figures saved to {FIG_DIR.resolve()}")


if __name__ == "__main__":
    main()
