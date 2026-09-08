"""线性回归用的合成数据集（PyTorch ``torch.utils.data.Dataset`` 风格）。

真实数据 ``Social_Network_Ads.xlsx`` 只有 2 个特征 + 二分类标签，没法直接
展示「线性回归」。这里用 ``y = 2·x1 + 3·x2 + 1 + ε`` 合成一份 200 条 2 维
样本，ground truth ``w=[2,3], b=1``，可以验证模型是否真的学到了线性系数。
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset


class LinearRegressionDataset(Dataset):
    """合成线性回归数据：``y = X @ w + b + ε``。"""

    def __init__(
        self,
        n_samples: int = 200,
        n_features: int = 2,
        noise: float = 0.1,
        true_w: tuple[float, ...] = (2.0, 3.0),
        true_b: float = 1.0,
        seed: int = 42,
    ) -> None:
        """
        Args:
            n_samples: 样本数。
            n_features: 特征维度。
            noise: 高斯噪声标准差。
            true_w: 真实权重向量（长度为 ``n_features``）。
            true_b: 真实偏置。
            seed: 随机种子，保证可复现。
        """
        if len(true_w) != n_features:
            raise ValueError(
                f"len(true_w)={len(true_w)} 与 n_features={n_features} 不一致"
            )

        g = torch.Generator().manual_seed(seed)
        self.features: torch.Tensor = torch.randn(n_samples, n_features, generator=g)
        w = torch.tensor(true_w, dtype=torch.float32).view(n_features, 1)
        noise_t = torch.randn(n_samples, 1, generator=g) * noise
        # y shape: (N, 1)
        self.targets: torch.Tensor = self.features @ w + true_b + noise_t
        # 记录 ground truth，方便训练后对照参数
        self.true_w: torch.Tensor = w
        self.true_b: float = true_b

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """返回单个样本 ``(features, target)``，target shape 为 ``(1,)``。"""
        return self.features[index], self.targets[index, 0]
