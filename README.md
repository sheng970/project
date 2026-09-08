# PyTorch 风格：线性回归 + 逻辑回归 + 7 个梯度下降优化器

完整覆盖图上的 5 条要求：
1. 线性回归 + 逻辑回归
2. 7 个梯度下降算法
3. 算法基本概念用类封装：`DataSet / Model / Loss / Optimizer`
4. 调用代码：大 `for`（epoch） + 小 `for`（batch）
5. 封装 + 调用风格参考 PyTorch 框架

## 目录结构

```
data/
├── classify/      # 逻辑回归数据集（Social_Network_Ads.xlsx → Dataset）
│   ├── dataset.py       SocialNetworkAdsDataset(Dataset)
│   └── features.py      切分/列名规范化工具
├── regression/    # 线性回归合成数据集
│   └── dataset.py       LinearRegressionDataset(Dataset)，y = 2x₁+3x₂+1+ε
├── model/         # 模型（nn.Module）
│   └── model.py         LinearRegression / LogisticRegression
├── loss/          # 损失（nn.Module）
│   └── loss.py          MSELoss / BCELoss
├── optim/         # 训练流程 + 7 个自实现 Optimizer
│   ├── optimizers.py    BGD / SGD / MBGD / Momentum / AdaGrad / RMSProp / Adam
│   └── optimizer.py     训练/评估/配置工具（保留旧版以兼容 one_hot_test.py）
├── raw/           # 原始数据
└── runs/          # 训练权重产物
main.py            # 端到端双流水线入口
one_hot_test.py    # 独热编码示例
requirements.txt
```

## 7 个梯度下降优化器

| 类 | 公式 | 调用方建议 |
|---|---|---|
| `BGD` Batch GD | `θ ← θ - lr·g` | `DataLoader(batch_size=N, shuffle=False)` 每 epoch 1 步 |
| `SGD` Stochastic GD | `θ ← θ - lr·g` | `DataLoader(batch_size=1)` 每 epoch N 步 |
| `MBGD` Mini-Batch GD | `θ ← θ - lr·g` | `DataLoader(batch_size=32)` 折中 |
| `Momentum` | `v ← βv + g; θ ← θ - lr·v` | β=0.9 |
| `AdaGrad` | `cache += g²; θ ← θ - lr·g/√(cache+ε)` | 适合稀疏特征 |
| `RMSProp` | `cache ← α·cache + (1-α)g²; θ ← θ - lr·g/√(cache+ε)` | α=0.9 |
| `Adam` | m/v 指数滑动平均 + 偏差修正 | 默认 β=(0.9, 0.999) |

三者更新公式一致，差异仅在调用方 `DataLoader` 的 `batch_size`。
三者更新公式一致，差异仅在调用方 `DataLoader` 的 `batch_size`。

## 训练循环（大 for + 小 for）

```python
optimizer = build_optimizer("Adam", model.parameters(), lr=0.01)
for epoch in range(epochs):                    # 大 for：epoch
    for x, y in dataloader:                    # 小 for：batch
        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()                        # autograd 算 param.grad
        optimizer.step()                       # 自定义 Optimizer 按公式更新
```

## 使用

```bash
pip install -r requirements.txt

# 默认：跑线性 + 逻辑两条流水线，Adam 优化器，200 epoch
python main.py

# 指定优化器（从 7 个中任选）
python main.py --optimizer MBGD
python main.py --optimizer Momentum
python main.py --optimizer AdaGrad

# 指定任务
python main.py --task linear
python main.py --task logistic

# 全参数示例
python main.py --task all --optimizer Adam --epochs 200 --batch-size 32 --lr 0.05
```

## 7 个优化器实测对比（逻辑回归，30 epoch）

| 优化器 | 准确率 |
|---|---|
| BGD | 0.8250 |
| SGD | 0.8125 |
| MBGD | 0.8250 |
| Momentum | 0.8625 |
| AdaGrad | 0.8375 |
| RMSProp | 0.8625 |
| **Adam** | **0.8625** |

## 关键设计

- **职责单一**：`Optimizer` 只做 `zero_grad` + 按公式更新。`param.grad` 由 `loss.backward()` 算
- **状态管理**：每个 `Optimizer` 内部用 `id(p)` 作 key 维护 `m` / `cache` / `_t` 等状态（Momentum/AdaGrad/RMSProp/Adam）
- **接口对齐 PyTorch**：`Optimizer(params, lr, ...)`、`zero_grad()`、`step()` 与 `torch.optim.Optimizer` 一致
- **数据集**：全部 `torch.utils.data.Dataset` 子类，可被 `DataLoader` 直接批量化
- **模型/损失**：全部 `torch.nn.Module` 子类，参数 `requires_grad=True` 走 autograd
