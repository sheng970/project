"""7 个梯度下降优化器（自实现，仿 ``torch.optim.Optimizer`` 接口风格）。

设计要点
========

1. **职责单一**：每个 ``Optimizer`` 只负责「清空梯度 (``zero_grad``) → 按各自公式
   更新参数 (``step``)」。``param.grad`` 由调用方的 ``loss.backward()`` 算出，
   优化器只是「读取梯度 + 套用更新公式」。这正是 PyTorch 官方 ``torch.optim``
   的设计。

2. **三组范式**：

   - 基础三件套（BGD / SGD / MBGD）：更新公式完全相同，差别只在调用方给
     ``DataLoader`` 传的 ``batch_size``——BGD 取整批、SGD 取 1、MBGD 取折中。
     这一组讲清「梯度下降的三种粒度」。
   - 动量系（Momentum）：引入速度项 ``v = βv + g``，解决 SGD 振荡。
   - 自适应系（AdaGrad / RMSProp / Adam）：每个参数独立维护学习率。

3. **使用模板（仿 PyTorch 风格）**::

       optimizer = Adam(model.parameters(), lr=0.01)
       for epoch in range(epochs):                       # 大 for (epoch)
           for x, y in dataloader:                        # 小 for (batch)
               optimizer.zero_grad()
               y_pred = model(x)
               loss = loss_fn(y_pred, y)
               loss.backward()
               optimizer.step()

4. **命名对照表**（与原论文/教材一致）：

   =============  ====================  =====================================
   类名           别名                  核心公式
   =============  ====================  =====================================
   ``BGD``        Batch GD              ``θ ← θ - lr·g``
   ``SGD``        Stochastic GD         ``θ ← θ - lr·g``（每次 1 个样本）
   ``MBGD``       Mini-Batch GD         ``θ ← θ - lr·g``（每次 batch）
   ``Momentum``   SGD + Momentum        ``v ← βv + g;  θ ← θ - lr·v``
   ``AdaGrad``    Adaptive Gradient     ``cache += g²;  θ ← θ - lr·g/(√cache+ε)``
   ``RMSProp``    Root Mean Square Prop ``cache ← α·cache + (1-α)·g²;
                                          θ ← θ - lr·g/(√cache+ε)``
   ``Adam``       Adaptive Moment Est.  ``m, v 指数滑动平均 + 偏差修正``
   =============  ====================  =====================================
"""

from __future__ import annotations

from typing import Iterable

import torch
from torch import Tensor
from torch.nn import Parameter


class Optimizer:
    """自定义优化器基类（仿 ``torch.optim.Optimizer``）。"""

    def __init__(self, params: Iterable[Parameter], lr: float) -> None:
        """缓存参数列表与每个参数对应的状态字典。

        Args:
            params: 模型可学习参数（通常传 ``model.parameters()``）。
            lr: 学习率。
        """
        self.params: list[Parameter] = [p for p in params if p.requires_grad]
        self.lr: float = lr
        # 用 id(p) 作为 key 维护每个参数的优化器状态（动量、cache、step 等）
        self._state: dict[int, dict[str, Tensor]] = {}

    # ------------------------------------------------------------------ API
    def zero_grad(self) -> None:
        """将所有 ``param.grad`` 置零（在每个 batch 训练开始时调用）。"""
        for p in self.params:
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

    def step(self) -> None:
        """按各自公式更新参数。子类必须实现。"""
        raise NotImplementedError

    # ------------------------------------------------------------- helpers
    def _state_for(self, p: Parameter) -> dict[str, Tensor]:
        """取出参数 ``p`` 对应的状态字典（惰性创建）。"""
        pid = id(p)
        if pid not in self._state:
            self._state[pid] = {}
        return self._state[pid]


# =====================================================================
# 基础三件套：更新公式相同，区别仅在调用方 DataLoader 的 batch_size
# =====================================================================


class BGD(Optimizer):
    """Batch Gradient Descent（整批梯度下降）。

    每 epoch 恰好 1 步：调用方应令 ``DataLoader(batch_size=len(dataset), shuffle=False)``。
    收敛稳定但每步代价高；样本量大时内存吃紧。
    """

    name = "BGD"

    def step(self) -> None:
        for p in self.params:
            if p.grad is None:
                continue
            p.data.add_(p.grad, alpha=-self.lr)


class SGD(Optimizer):
    """Stochastic Gradient Descent（随机梯度下降）。

    每 epoch 走 N 步：调用方应令 ``DataLoader(batch_size=1)``。每步更新噪声大，
    但能跳出鞍点、对在线学习友好。
    """

    name = "SGD"

    def step(self) -> None:
        for p in self.params:
            if p.grad is None:
                continue
            p.data.add_(p.grad, alpha=-self.lr)


class MBGD(Optimizer):
    """Mini-Batch Gradient Descent（小批量梯度下降）。

    每 epoch 走 ``ceil(N / batch_size)`` 步：调用方令 ``DataLoader(batch_size=32)``
    （或类似）。是 BGD 与 SGD 的折中，深度学习事实标准。
    """

    name = "MBGD"

    def step(self) -> None:
        for p in self.params:
            if p.grad is None:
                continue
            p.data.add_(p.grad, alpha=-self.lr)


# =====================================================================
# 动量系：在 SGD 基础上引入「速度项」平滑轨迹
# =====================================================================


class Momentum(Optimizer):
    """SGD with Momentum（带动量的随机梯度下降）。

    公式：``v ← β·v + g;  θ ← θ - lr·v``。相当于把梯度按指数滑动平均后
    再更新，能在「山谷」「峡谷」地形中加速收敛并抑制振荡。
    """

    name = "Momentum"

    def __init__(self, params: Iterable[Parameter], lr: float, momentum: float = 0.9) -> None:
        super().__init__(params, lr)
        self.momentum = momentum

    def step(self) -> None:
        for p in self.params:
            if p.grad is None:
                continue
            state = self._state_for(p)
            v = state.setdefault("v", torch.zeros_like(p.data))
            # v = β·v + g
            v.mul_(self.momentum).add_(p.grad)
            # θ -= lr·v
            p.data.add_(v, alpha=-self.lr)


# =====================================================================
# 自适应系：每个参数独立缩放学习率
# =====================================================================


class AdaGrad(Optimizer):
    """AdaGrad（Adaptive Gradient，自适应梯度）。

    公式：``cache += g²;  θ -= lr·g / (√cache + ε)``。
    频繁更新的参数学习率被压低、稀疏参数保留较大步长；缺点是 cache 单调
    累积，后期学习率过小导致提前收敛。
    """

    name = "AdaGrad"

    def __init__(self, params: Iterable[Parameter], lr: float, eps: float = 1e-8) -> None:
        super().__init__(params, lr)
        self.eps = eps

    def step(self) -> None:
        for p in self.params:
            if p.grad is None:
                continue
            state = self._state_for(p)
            cache = state.setdefault("cache", torch.zeros_like(p.data))
            # cache += g²
            cache.addcmul_(p.grad, p.grad, value=1.0)
            # θ -= lr·g / (√cache + ε)
            p.data.addcdiv_(p.grad, cache.sqrt().add_(self.eps), value=-self.lr)


class RMSProp(Optimizer):
    """RMSProp（Root Mean Square Propagation）。

    公式：``cache ← α·cache + (1-α)·g²;  θ -= lr·g / (√cache + ε)``。
    用指数滑动平均替代 AdaGrad 的单调累加，让 cache「遗忘」久远梯度，
    解决 AdaGrad 后期学习率过小的问题。
    """

    name = "RMSProp"

    def __init__(
        self,
        params: Iterable[Parameter],
        lr: float,
        alpha: float = 0.9,
        eps: float = 1e-8,
    ) -> None:
        super().__init__(params, lr)
        self.alpha = alpha
        self.eps = eps

    def step(self) -> None:
        for p in self.params:
            if p.grad is None:
                continue
            state = self._state_for(p)
            cache = state.setdefault("cache", torch.zeros_like(p.data))
            # cache ← α·cache + (1-α)·g²
            cache.mul_(self.alpha).addcmul_(p.grad, p.grad, value=1.0 - self.alpha)
            # θ -= lr·g / (√cache + ε)
            p.data.addcdiv_(p.grad, cache.sqrt().add_(self.eps), value=-self.lr)


class Adam(Optimizer):
    """Adam（Adaptive Moment Estimation，自适应矩估计）。

    同时维护一阶矩 ``m``（梯度的指数滑动平均，类似 Momentum）和二阶矩
    ``v``（梯度平方的指数滑动平均，类似 RMSProp），并对前几步做偏差修正：

        m  ← β1·m + (1-β1)·g
        v  ← β2·v + (1-β2)·g²
        m̂  = m / (1 - β1^t)
        v̂  = v / (1 - β2^t)
        θ  ← θ - lr·m̂ / (√v̂ + ε)
    """

    name = "Adam"

    def __init__(
        self,
        params: Iterable[Parameter],
        lr: float,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ) -> None:
        super().__init__(params, lr)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self._t: int = 0  # 全局步数（所有参数共享，PyTorch 也按此实现）

    def step(self) -> None:
        self._t += 1
        for p in self.params:
            if p.grad is None:
                continue
            state = self._state_for(p)
            m = state.setdefault("m", torch.zeros_like(p.data))
            v = state.setdefault("v", torch.zeros_like(p.data))
            # m ← β1·m + (1-β1)·g
            m.mul_(self.beta1).add_(p.grad, alpha=1.0 - self.beta1)
            # v ← β2·v + (1-β2)·g²
            v.mul_(self.beta2).addcmul_(p.grad, p.grad, value=1.0 - self.beta2)
            # 偏差修正
            bc1 = 1.0 - self.beta1 ** self._t
            bc2 = 1.0 - self.beta2 ** self._t
            m_hat = m / bc1
            v_hat = v / bc2
            # θ -= lr·m̂ / (√v̂ + ε)
            p.data.addcdiv_(m_hat, v_hat.sqrt().add_(self.eps), value=-self.lr)


# ------------------------------------------------------------------ 工厂
def build_optimizer(
    name: str,
    params: Iterable[Parameter],
    lr: float,
    **kwargs,
) -> Optimizer:
    """按名字构造优化器（与 ``torch.optim`` 的 ``get`` 风格类似）。"""
    registry: dict[str, type[Optimizer]] = {
        cls.name: cls for cls in (BGD, SGD, MBGD, Momentum, AdaGrad, RMSProp, Adam)
    }
    if name not in registry:
        raise ValueError(
            f"unknown optimizer '{name}', "
            f"available: {sorted(registry.keys())}"
        )
    return registry[name](params, lr=lr, **kwargs)
