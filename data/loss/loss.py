"""损失函数集合（PyTorch ``nn.Module`` 风格）。

包含：
- ``MSELoss``：均方误差，用于回归任务（线性回归）
- ``BCELoss``：二分类交叉熵，用于二分类任务（逻辑回归）
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class MSELoss(nn.Module):
    """均方误差损失（Mean Squared Error）。

    公式：``MSE = mean((y_pred - y_true)²)``。
    常用于回归任务，对异常值敏感（因为有平方放大）。
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """计算 MSE。

        Args:
            y_pred: 模型预测，shape 与 ``y_true`` 一致。
            y_true: 真实值。

        Returns:
            标量损失张量。
        """
        diff = y_pred - y_true
        return (diff * diff).mean()


class BCELoss(nn.Module):
    """二分类交叉熵损失（Binary Cross Entropy）。

    公式：``BCE = -mean(y·log(p) + (1-y)·log(1-p))``，
    ``p`` 经 ``clamp`` 防止 ``log(0)``。
    """

    def __init__(self, eps: float = 1e-12) -> None:
        super().__init__()
        self._eps = eps

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """计算 BCE 损失。

        Args:
            y_pred: 模型预测概率 (0~1)，与 ``y_true`` 形状一致。
            y_true: 真实标签 (0/1)。

        Returns:
            标量损失张量。
        """
        y_pred = torch.clamp(y_pred, min=self._eps, max=1.0 - self._eps)
        return -(y_true * torch.log(y_pred) + (1.0 - y_true) * torch.log(1.0 - y_pred)).mean()


def binary_cross_entropy(y_true: Tensor, y_pred: Tensor, eps: float = 1e-12) -> Tensor:
    """函数式 BCE，便于一次性评估。"""
    return BCELoss(eps=eps)(y_pred, y_true)
