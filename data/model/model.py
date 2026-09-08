"""模型定义（PyTorch ``nn.Module`` 风格）。

包含两个模型：
- ``LinearRegression``：单层 ``nn.Linear``，输出连续值（用于回归任务）
- ``LogisticRegression``：单层 ``nn.Linear`` + Sigmoid，输出 0~1 概率（用于二分类）

风格与 PyTorch 官方 ``nn.Module`` 完全一致：
- ``__init__`` 定义子层
- ``forward`` 描述数据流
- ``save / load`` 用 ``state_dict`` 落盘与重载
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor


class LinearRegression(nn.Module):
    """线性回归模型：``y = W x + b``（单层线性，无激活）。"""

    def __init__(self, input_dim: int, name: str = "linear-regression") -> None:
        super().__init__()
        self.name = name
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x: Tensor) -> Tensor:
        """前向：``y = W x + b``，输出 shape ``(N, 1)``。"""
        return self.linear(x)

    def save(self, path: Path) -> None:
        """保存 state_dict 到磁盘。"""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    def load(self, path: Path) -> "LinearRegression":
        """从 state_dict 恢复参数。"""
        self.load_state_dict(torch.load(path, map_location="cpu"))
        return self


class LogisticRegression(nn.Module):
    """逻辑回归二分类器（单层线性 + Sigmoid）。"""

    def __init__(self, input_dim: int, name: str = "logistic-regression") -> None:
        super().__init__()
        self.name = name
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x: Tensor) -> Tensor:
        """前向：``y = sigmoid(W x + b)``，输出 0~1 概率。"""
        return torch.sigmoid(self.linear(x))

    def save(self, path: Path) -> None:
        """保存 state_dict 到磁盘。"""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    def load(self, path: Path) -> "LogisticRegression":
        """从 state_dict 恢复参数。"""
        self.load_state_dict(torch.load(path, map_location="cpu"))
        return self
