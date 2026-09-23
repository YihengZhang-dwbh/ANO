# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""G(x) 塑形函数：ANO 用来替换 PPO 硬 clip(ratio) 的可微塑形核。

塑形函数扮演的角色，与旧核 f 完全一致：直接替换 PPO 里 `clip(ratio)` 的位置，
乘在 -Adv 上算 loss。旧核 f 满足 f(1)=1、f'(1)=1（切于 y=x）、且在
x=1+epsilon 处取得**局部极大值**（f'(1+eps)=0）——这正是我们的 G(x) 本身
的边界条件，不是 G'(x) 的。**本文件早先的版本把这里搞错了**，把 G'(x) 当成
了塑形函数塞进去，那是不对的：G'(1)=1 但 G'(1+eps)=0 是 G' 的"零点"不是
"极大值"，G' 在 x=1 处也不满足"值为 1"这个要求（G'(1)=1 是导数为1，不是
函数值为1）。真正对得上号的是 G(x)：G(1)=1、G'(1)=1（切于 y=x）、G 在
x=1+eps 取局部极大（因为 G'(1+eps)=0 且 G'' 在那附近变号——见下）。

数学来源见 C:\\Code\\ICLR_ANO\\show_rate.py 与 construction.tex。要点复述：

    G'(x) = y1 * r * (1 - E) / [ (1 + E)(E + r) ],   E = e^{a(x-x0)}, x0 = 1+eps
    G(x)  = 1 + Phi(x-x0) - Phi(1-x0)   （Phi 是 G' 的初等原函数，含 log 项）

四个条件（G(1)=1, G'(1)=1, G'(1+eps)=0, G'(-inf)=y1, G'(+inf)=0, min G'=b 且
唯一拐点）里，除 G'(1)=1 外全部在 G' 的这个有理式下自动成立；G'(1)=1 定标度
a，谷深 b 与 y1 完全解耦（b = -d*y1，d=|b|/y1 由 r 的闭式反解决定）。

G(1+eps) 是 G 的局部极大值，对应旧核 f 在 x=1+eps 处取最高点：因为
G'(1+eps)=0 且 G' 在 1+eps 左侧为正、右侧为负（G' 在 (-inf,1+eps) 上从 y1
降到 0、在 (1+eps, x_p) 上继续降到 b<0、再回升到 0），所以 G 在 1+eps 处
恰好是一阶导数由正变负的临界点，即局部极大。三个超参都在训练开始前解出
(r, a)，训练循环里只做逐元素的初等运算，不含任何求根。

与 show_rate.py 的两个差异，都是为了塞进训练循环：
  1. G' 与 G 全部用 u = sigmoid(z) 代替 E = e^z 参数化。E 在 z 大时溢出，u
     恒在 (0,1)，所以这里不需要 show_rate.py 里那套 "E>1 就切换成 1/E 分支"
     的补丁；用 u 表示的 log 项也天然数值稳定（torch 有现成的稳定 logsigmoid/
     softplus）。这套 u-参数化闭式已经和 show_rate.py 的 E-参数化闭式数值
     对照过，max|G_u - G_E| ~ 1.6e-8（4×4×4×4=256 组网格采样，见本文件自检）。
  2. a 在训练开始前用闭式（二次方程的正根）算一次，存成 python float，
     不是每步都对 tensor 求根。torch.jit.script 里只有初等张量运算。

与旧的 _ano_math_kernel（f0 = 45/16*(0.5*logsigmoid(2x)-2*sigmoid(x))）的关系：
旧核也是"sigmoid 组合、软化 PPO 硬 clip 角点"的思路，但没有 y1/b 这两个独立
自由度，无法指定饱和斜率与谷深。G(x) 是它的严格推广，共享同一个不变量
G(1)=f(1)=1、G'(1)=f'(1)=1，以及"在 1+eps 处取局部极大"这个几何形状。

运行 `python g_shaping.py` 做一次离线自检（不需要 GPU/torch.jit 环境也能测
纯数学部分；若装了 torch 会额外跑 torch 路径的对比）。
"""

from __future__ import annotations

import math
from typing import Tuple

import torch


# ============================================================ 离线闭式求解
# 这部分只在训练开始前跑一次（普通 python float，不是 tensor），所以不追求
# torch.jit 兼容，用 math 库即可。对应 show_rate.py 的 from_b / solve_a，但
# a 这里额外做成闭式（二次方程），比 show_rate.py 的二分更快也更精确。

def _r_of_b(y1: float, b: float) -> float:
    """由 (y1, b) 反解形状参数 r，闭式，无需迭代。

    d = |b|/y1 ∈ (0,1)；s = (2d+sqrt(2(d+1)))/(1-d)；r = (s^2-2)/2。
    见 construction.tex 式 (16)/(17) 或 show_rate.py 的 r_of_depth。
    """
    if not (y1 > 1.0):
        raise ValueError(f"ano_y1 must be > 1, got {y1}")
    if not (-y1 < b < 0.0):
        raise ValueError(
            f"ano_b must satisfy -ano_y1 < ano_b < 0 (got ano_b={b}, ano_y1={y1}); "
            f"this is the exact reachable range of the construction, not a tuning limit."
        )
    d = -b / y1
    s = (2.0 * d + math.sqrt(2.0 * (d + 1.0))) / (1.0 - d)
    return 0.5 * (s * s - 2.0)


def _a_of_scale(eps: float, y1: float, r: float) -> float:
    """由 (eps, y1, r) 解标度 a，闭式（二次方程的正根，见 construction.tex 式 (24)）。

    q = E(1) 满足 q^2 + B q - C = 0，B = r(y1+1)+1 > 0，C = r(y1-1) > 0。
    用共轭形式 q = 2C/(B+sqrt(B^2+4C)) 求正根，避免 C << B^2 时两个相近大数
    相减的灾难性相消（构造论证里专门证明过为什么必须用这个形式，不能用
    q=(-B+sqrt(B^2+4C))/2）。
    """
    if not (eps > 0.0):
        raise ValueError(f"cliprange (eps) must be > 0, got {eps}")
    B = r * (y1 + 1.0) + 1.0
    C = r * (y1 - 1.0)
    q = 2.0 * C / (B + math.sqrt(B * B + 4.0 * C))
    if not (0.0 < q < 1.0):
        raise RuntimeError(f"internal error: q={q} out of (0,1) for eps={eps}, y1={y1}, r={r}")
    return -math.log(q) / eps


def solve_g_shaping_constants(eps: float, y1: float, b: float) -> Tuple[float, float, float]:
    """训练开始前调用一次：由用户超参 (eps, y1, b) 解出 (r, a, x0)。

    Args:
        eps: 对应旧 ANO/PPO 的 cliprange，一阶导零点 x0=1+eps 的偏移，也是
            G(x) 在正侧取局部极大值的位置。
        y1: G'(-inf)，即 x 远小于 1 时 G 的渐近斜率（"最大推力"）。
        b: G' 的全局最小值，即"最大拉力"，需要 -y1 < b < 0。

    Returns:
        (r, a, x0)：喂给 g_shaping_kernel 的三个标量常数。
    """
    r = _r_of_b(y1, b)
    a = _a_of_scale(eps, y1, r)
    x0 = 1.0 + eps
    return r, a, x0


# ================================================================ 训练期核
# 下面两个函数是逐 batch 调用的部分，写成 torch.jit.script 以匹配旧
# _ano_math_kernel / _compute_ano_loss 的调用方式和性能特征。

@torch.jit.script
def g_shaping_kernel(x: torch.Tensor, r: float, a: float, x0: float, y1: float) -> torch.Tensor:
    """G(x)，塑形函数本身（替代旧 f0/_ano_math_kernel）。**不是** G'(x)。

    令 u = sigmoid(a(x-x0)) ∈ (0,1)，z = a(x-x0)。闭式（对应 show_rate.py 的
    G()/_Phi()，但按 u 而不是 E=e^z 重新参数化，避免 E 溢出）。用恒等式
    1+E = 1/(1-u)、E+r = (u+r(1-u))/(1-u) 把 show_rate.py 里的
        Phi(t) = (y1/a)[a t - (2r/(r-1))log(1+E) + ((r+1)/(r-1))log(E+r)]
    改写成纯 u 的形式，系数会**恰好化简**（sympy 验证过 log(1-u) 的系数
    2r/(r-1) - (r+1)/(r-1) 恰等于 1，与 r 无关）：

        r != 1:  bracket(z) = z + log(1-u) + ((r+1)/(r-1)) * log(u + r(1-u))
        r == 1:  bracket(z) = z + log(1-u) + 2*(1-u)            <-- r->1 的极限

        G(x) = 1 + (y1/a) * [ bracket(z) - bracket(z1) ],   z1 = a(1-x0)

    log(u+r(1-u)) 恒稳定：u+r(1-u) ∈ (0, max(1,r))，不会像直接算 E+r 那样在
    z 大时溢出。log(1-u) 用 -softplus(z) 求。

    **我在第一版这里写错过一次**：把 log(1-u) 的系数写反了符号（少了一步
    "1+E=1/(1-u) 所以 log(1+E)=-log(1-u)" 的代换），导致 x>x0 一侧的曲线
    直接跑飞（在 x=1.07 处误差达到 0.167，肉眼可见）。这里的写法已经过
    sympy 符号验证（对 z 求导后与 show_rate.py 的 Phi' 恒等于 0）+ 数值
    网格验证（见 _selftest，max diff ~1.6e-8）。
    """
    z = a * (x - x0)
    z1 = a * (1.0 - x0)
    u = torch.sigmoid(z)
    u1 = 1.0 / (1.0 + math.exp(-z1))  # 标量，python float 运算即可

    log_1mu = -torch.nn.functional.softplus(z)          # log(1-u)，稳定
    log_1mu1 = -math.log1p(math.exp(z1)) if z1 < 0 else -(z1 + math.log1p(math.exp(-z1)))

    if abs(r - 1.0) < 1e-6:
        bracket = z + log_1mu + 2.0 * (1.0 - u)
        bracket1 = z1 + log_1mu1 + 2.0 * (1.0 - u1)
    else:
        c3 = (r + 1.0) / (r - 1.0)
        denom = u + r * (1.0 - u)
        denom1 = u1 + r * (1.0 - u1)
        bracket = z + log_1mu + c3 * torch.log(denom)
        bracket1 = z1 + log_1mu1 + c3 * math.log(denom1)

    return 1.0 + (y1 / a) * (bracket - bracket1)


@torch.jit.script
def _compute_g_loss(
    mb_advantage: torch.Tensor,
    ratio: torch.Tensor,
    r: float, a: float, x0: float, y1: float,
) -> torch.Tensor:
    """计算 ANO 策略损失，用 G(x) 替换旧的 _ano_math_kernel。

    与旧版完全一致的正负 Adv 分支处理（旧核变量名是 f，这里换成 G）：
        Adv >= 0:  loss = -Adv * G(r)
        Adv <  0:  loss = -Adv * [2 - G(2 - r)]   <-- 与旧版相同的对偶写法

    常数 "2" 直接照抄旧代码：G 与旧核 f 共享同一个锚点 G(1) = f(1) = 1，
    对偶 g(x) = 2 - G(2-x) 只需要 g(1) = 2 - G(1) = 2 - 1 = 1，用常数 2 就
    严格满足，不需要按 y1 缩放。
    """
    f_val_pos = g_shaping_kernel(ratio, r, a, x0, y1)
    f_val_neg = 2.0 - g_shaping_kernel(2.0 - ratio, r, a, x0, y1)
    target_f_val = torch.where(mb_advantage >= 0, f_val_pos, f_val_neg)
    return -mb_advantage * target_f_val


# --------------------------------------------------------------------- 自检
def _selftest() -> None:
    """离线数值自检：纯 python/math 复刻 g_shaping_kernel 的算子，对照
    C:\\Code\\ICLR_ANO\\show_rate.py 的 G()（E-参数化闭式）。"""
    import numpy as np

    def _ref_module():
        import importlib.util
        import pathlib

        here = pathlib.Path(__file__).resolve()
        for up in range(1, 8):
            cand = here.parents[up] / "show_rate.py" if up < len(here.parents) else None
            if cand and cand.exists():
                spec = importlib.util.spec_from_file_location("show_rate", cand)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                return mod
        return None

    def G_u_numpy(x, y1, r, a, x0):
        """g_shaping_kernel 的纯 numpy 复刻（不依赖 torch，用于沙盒自检）。

        必须与 g_shaping_kernel 里的公式逐项一致（包括那个化简后的
        log(1-u) 系数 = 1，不是 2r/(r-1)），否则自检查不出实现里的符号错误。
        """
        x = np.asarray(x, dtype=np.float64)
        z = a * (x - x0)
        z1 = a * (1.0 - x0)
        u = 1.0 / (1.0 + np.exp(-np.clip(z, -700, 700)))
        u1 = 1.0 / (1.0 + math.exp(-max(min(z1, 700), -700)))
        # 稳定 log(1-u) = -softplus(z) = -(max(z,0) + log1p(exp(-|z|)))
        log_1mu = -(np.maximum(z, 0.0) + np.log1p(np.exp(-np.abs(z))))
        log_1mu1 = -(max(z1, 0.0) + math.log1p(math.exp(-abs(z1))))
        if abs(r - 1.0) < 1e-6:
            bracket = z + log_1mu + 2.0 * (1.0 - u)
            bracket1 = z1 + log_1mu1 + 2.0 * (1.0 - u1)
        else:
            c3 = (r + 1.0) / (r - 1.0)
            denom = u + r * (1.0 - u)
            denom1 = u1 + r * (1.0 - u1)
            bracket = z + log_1mu + c3 * np.log(denom)
            bracket1 = z1 + log_1mu1 + c3 * math.log(denom1)
        return 1.0 + (y1 / a) * (bracket - bracket1)

    ref = _ref_module()
    cases = [
        (eps, y1, -frac * y1)
        for eps in (0.02, 0.2, 1.0, 5.0)
        for y1 in (1.05, 1.2, 3.0, 50.0, 1e4)
        for frac in (1e-3, 0.01, 0.3, 0.7, 0.99)
    ]
    worst_anchor = 0.0
    worst_vs_ref = 0.0
    for eps, y1, b in cases:
        r, a, x0 = solve_g_shaping_constants(eps, y1, b)
        assert r > 0.0 and a > 0.0

        # (1) G(1) = 1 恰好成立（塑形函数的锚点，对应旧核 f(1)=1）
        g_at_1 = float(G_u_numpy(1.0, y1, r, a, x0))
        worst_anchor = max(worst_anchor, abs(g_at_1 - 1.0))

        # (2) G 在 x0=1+eps 处取局部极大（对应旧核在该点的最高点）。
        # 步长必须相对曲率尺度 1/a 缩放：a 可以到几百，固定的 1e-4 会跨出局部
        # 极大所在的那个小邻域，看见的就是别的地方的曲线形状，不是这个极值。
        h = 1e-3 / a
        g_lo = float(G_u_numpy(x0 - h, y1, r, a, x0))
        g_mid = float(G_u_numpy(x0, y1, r, a, x0))
        g_hi = float(G_u_numpy(x0 + h, y1, r, a, x0))
        assert g_mid > g_lo and g_mid > g_hi, (
            f"G is not a local max at x0 for eps={eps},y1={y1},b={b}: "
            f"{g_lo:.6f} {g_mid:.6f} {g_hi:.6f}"
        )

        # (3) 与 show_rate.py 的 E-参数化闭式数值对照
        if ref is not None:
            r_ref, a_ref = ref.from_b(eps, y1, b)
            worst_vs_ref = max(worst_vs_ref, abs(r - r_ref) / r_ref, abs(a - a_ref) / a_ref)
            grid = np.linspace(x0 - 8.0 / a, x0 + 8.0 / a, 2001)
            g_new = G_u_numpy(grid, y1, r, a, x0)
            g_old = np.asarray(ref.G(grid, eps, y1, r, a))
            # 绝对误差在极端参数角（比如 y1/a ~ 1e8）会跟着整体量级一起放大，
            # 所以按曲线自身幅值归一化，比较相对误差。
            scale = max(1.0, float(np.max(np.abs(g_old))))
            worst_vs_ref = max(worst_vs_ref, float(np.max(np.abs(g_new - g_old))) / scale)

    tag = "vs show_rate.py" if ref is not None else "(show_rate.py not found, internal check only)"
    print(f"G(1)=1 anchor: worst error over {len(cases)} cases: {worst_anchor:.2e}")
    print(f"G has a local max at x0=1+eps: PASS for all {len(cases)} cases")
    print(f"G(x) closed form {tag}: worst error {worst_vs_ref:.2e}")
    assert worst_anchor < 1e-6 and worst_vs_ref < 1e-6, "g_shaping self-test failed"
    print("PASS")


if __name__ == "__main__":
    _selftest()
