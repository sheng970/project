"""特征工程工具。"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def to_snake_case(name: str) -> str:
    """将字符串转成小写蛇形命名（支持驼峰与空格）。"""
    stepped = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    stepped = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", stepped)
    stepped = re.sub(r"[\s]+", "_", stepped).lower()
    return re.sub(r"_+", "_", stepped).strip("_")


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """将 DataFrame 列名规范为小写蛇形。"""
    return df.rename(
        columns={c: to_snake_case(c) if isinstance(c, str) else c for c in df.columns}
    )


def split_features_target(
    df: pd.DataFrame,
    target: str,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """将 DataFrame 拆分为特征矩阵与目标向量。"""
    feature_cols = [c for c in df.columns if c != target]
    x = df[feature_cols].to_numpy(dtype=np.float64)
    y = df[target].to_numpy(dtype=np.float64)
    return x, y


def train_test_split_indices(
    n_samples: int,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """生成确定性训练/测试集下标（与 sklearn 切分风格一致）。"""
    rng = np.random.default_rng(random_state)
    perm = rng.permutation(n_samples)
    n_test = int(round(n_samples * test_size))
    return perm[n_test:], perm[:n_test]
