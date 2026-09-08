"""数据集加载（PyTorch ``torch.utils.data.Dataset`` 风格）。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from data.classify.features import clean_column_names

FILE_NAME = "Social_Network_Ads.xlsx"
TARGET_COLUMN = "purchased"


class SocialNetworkAdsDataset(Dataset):
    """社交网络广告二分类数据集。

    构造时即把 ``Age`` / ``EstimatedSalary`` 加载为 ``torch.float32`` 张量，
    并将 ``Purchased`` 作为 ``torch.float32`` 目标张量。
    满足 ``torch.utils.data.Dataset`` 协议，可被 :class:`torch.utils.data.DataLoader`
    直接批量化加载。
    """

    def __init__(self, data_path: Path) -> None:
        """加载 Excel 并构造样本张量。

        Args:
            data_path: Excel 文件路径或所在目录。

        Raises:
            FileNotFoundError: 文件不存在时抛出。
        """
        file_path = data_path / FILE_NAME if data_path.is_dir() else data_path
        if not file_path.exists():
            raise FileNotFoundError(f"data file not found: {file_path}")

        df = clean_column_names(pd.read_excel(file_path))
        feature_cols = [c for c in df.columns if c != TARGET_COLUMN]
        self.features: torch.Tensor = torch.tensor(
            df[feature_cols].to_numpy(dtype=np.float32),
        )
        self.targets: torch.Tensor = torch.tensor(
            df[TARGET_COLUMN].to_numpy(dtype=np.float32),
        )
        self.feature_names: list[str] = feature_cols

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """返回单个样本 (features, target)。"""
        return self.features[index], self.targets[index]
