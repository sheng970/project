"""独热编码示例脚本。"""

from __future__ import annotations

import numpy as np


def to_one_hot(values, classes=None) -> np.ndarray:
    """将分类值转成 one-hot 矩阵。

    Args:
        values: 任意可迭代的分类值。
        classes: 显式类别列表；为空则从 values 中按出现顺序推导。

    Returns:
        shape=(len(values), len(classes)) 的 one-hot 矩阵。
    """
    if classes is None:
        seen: list = []
        for v in values:
            if v not in seen:
                seen.append(v)
        classes = seen
    class_index = {c: i for i, c in enumerate(classes)}
    matrix = np.zeros((len(values), len(classes)), dtype=np.int8)
    for i, v in enumerate(values):
        matrix[i, class_index[v]] = 1
    return matrix


if __name__ == "__main__":
    sample = ["cat", "dog", "cat", "bird"]
    print(f"classes (sorted): {sorted(set(sample))}")
    print("one-hot:")
    print(to_one_hot(sample, classes=sorted(set(sample))))
