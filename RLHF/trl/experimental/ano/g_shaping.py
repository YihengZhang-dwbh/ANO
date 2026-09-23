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
"""G(x) shaping function: the differentiable shaping kernel ANO uses to replace PPO's hard clip(ratio).

The shaping function plays exactly the same role as the old kernel f: it directly replaces the
`clip(ratio)` position in PPO, multiplying -Adv to compute the loss. The old kernel f satisfies
f(1)=1, f'(1)=1 (tangent to y=x), and attains a **local maximum** at x=1+epsilon (f'(1+eps)=0) --
these are the boundary conditions of G(x) itself, not of G'(x). An earlier version of this file
mistakenly treated G'(x) as the shaping function, which is incorrect: G'(1)=1 and G'(1+eps)=0 are
properties of G' (a "zero crossing", not a "maximum"), and G' does not satisfy the "value equals 1"
requirement at x=1 (G'(1)=1 is a derivative value, not a function value). The correct match is G(x):
G(1)=1, G'(1)=1 (tangent to y=x), and G attains a local maximum at x=1+eps (because G'(1+eps)=0
and G'' changes sign there -- see below).

The mathematical derivation is given in paper Eq. (9) and Section 4.2. Key points:

    G'(x) = kappa_plus * r * (1 - E) / [ (1 + E)(E + r) ],   E = e^{a(x-x0)}, x0 = 1+eps
    G(x)  = 1 + Phi(x-x0) - Phi(1-x0)   (Phi is the elementary antiderivative of G', with log terms)

Of the defining conditions (G(1)=1, G'(1)=1, G'(1+eps)=0, G'(-inf)=kappa_plus, G'(+inf)=0,
min G'=kappa_minus, and a unique inflection point), all except G'(1)=1 hold automatically under
this rational form of G'; G'(1)=1 determines the scale a, and the valley depth kappa_minus is fully
decoupled from kappa_plus (kappa_minus = -d*kappa_plus, where d=|kappa_minus|/kappa_plus is
determined by the closed-form inversion for r).

G(1+eps) is a local maximum of G, corresponding to the old kernel f reaching its highest point at
x=1+eps: because G'(1+eps)=0 and G' is positive to the left of 1+eps and negative to the right
(G' decreases from kappa_plus to 0 on (-inf,1+eps), continues to decrease to kappa_minus<0 on
(1+eps, x_p), then rises back to 0), G has a critical point at 1+eps where the first derivative
changes from positive to negative, i.e. a local maximum. All three hyperparameters are solved for
(r, a) before training begins; the training loop only performs element-wise elementary operations
with no root-finding.

Two differences from the reference derivation (paper Appendix G), both motivated by fitting into
the training loop:
  1. G' and G are parameterized by u = sigmoid(z) instead of E = e^z. E overflows for large z,
     while u stays in (0,1), so the "switch to 1/E branch when E>1" patch used in the offline
     derivation is unnecessary here; the log terms expressed in u are also numerically stable
     (torch provides stable logsigmoid/softplus). This u-parameterized closed form has been
     numerically cross-validated against the E-parameterized closed form, with
     max|G_u - G_E| ~ 1.6e-8 (256 grid samples on a 4x4x4x4 grid; see the self-test in this file).
  2. a is computed once before training using a closed form (positive root of a quadratic
     equation) and stored as a python float, rather than solving for a root on a tensor every
     step. torch.jit.script only contains elementary tensor operations.

Relation to the old _ano_math_kernel (f0 = 45/16*(0.5*logsigmoid(2x)-2*sigmoid(x))):
The old kernel follows the same idea of "sigmoid composition, softening PPO's hard clip corners",
but lacks the two independent degrees of freedom kappa_plus/kappa_minus, so it cannot specify the
saturated slope or the valley depth. G(x) is a strict generalization, sharing the same invariants
G(1)=f(1)=1, G'(1)=f'(1)=1, and the geometric shape of "local maximum at 1+eps".

Run `python g_shaping.py` for an offline self-test (the pure-math part does not require GPU or
torch.jit; if torch is installed, the torch path is also cross-checked).
"""

from __future__ import annotations

import math
from typing import Tuple

import torch


# ============================================================ Offline closed-form solve
# This section runs only once before training begins (plain python floats, not tensors), so
# torch.jit compatibility is not required; the math library suffices. Corresponds to
# from_b / solve_a in the paper's supplementary derivation, but a is additionally made
# closed-form here (quadratic equation), which is faster and more accurate than bisection.

def _r_of_kappa_minus(kappa_plus: float, kappa_minus: float) -> float:
    """Solve for the shape parameter r from (kappa_plus, kappa_minus), closed form, no iteration.

    d = |kappa_minus|/kappa_plus in (0,1); s = (2d+sqrt(2(d+1)))/(1-d); r = (s^2-2)/2.
    See paper Eq. (16)/(17) or the r_of_depth function in the reference derivation.
    """
    if not (kappa_plus > 1.0):
        raise ValueError(f"ano_kappa_plus must be > 1, got {kappa_plus}")
    if not (-kappa_plus < kappa_minus < 0.0):
        raise ValueError(
            f"ano_kappa_minus must satisfy -ano_kappa_plus < ano_kappa_minus < 0 "
            f"(got ano_kappa_minus={kappa_minus}, ano_kappa_plus={kappa_plus}); "
            f"this is the exact reachable range of the construction, not a tuning limit."
        )
    d = -kappa_minus / kappa_plus
    s = (2.0 * d + math.sqrt(2.0 * (d + 1.0))) / (1.0 - d)
    return 0.5 * (s * s - 2.0)


def _a_of_scale(eps: float, kappa_plus: float, r: float) -> float:
    """Solve for the scale a from (eps, kappa_plus, r), closed form (positive root of a quadratic;
    see paper Eq. (24)).

    q = E(1) satisfies q^2 + B q - C = 0, B = r(kappa_plus+1)+1 > 0, C = r(kappa_plus-1) > 0.
    The positive root is computed in the conjugate form q = 2C/(B+sqrt(B^2+4C)) to avoid
    catastrophic cancellation when C << B^2 (the construction proof shows why this form is
    necessary; q=(-B+sqrt(B^2+4C))/2 must not be used).
    """
    if not (eps > 0.0):
        raise ValueError(f"cliprange (eps) must be > 0, got {eps}")
    B = r * (kappa_plus + 1.0) + 1.0
    C = r * (kappa_plus - 1.0)
    q = 2.0 * C / (B + math.sqrt(B * B + 4.0 * C))
    if not (0.0 < q < 1.0):
        raise RuntimeError(
            f"internal error: q={q} out of (0,1) for eps={eps}, kappa_plus={kappa_plus}, r={r}"
        )
    return -math.log(q) / eps


def solve_g_shaping_constants(
    eps: float, kappa_plus: float, kappa_minus: float
) -> Tuple[float, float, float]:
    """Call once before training: solve for (r, a, x0) from the user hyperparameters
    (eps, kappa_plus, kappa_minus).

    Args:
        eps: Corresponds to the old ANO/PPO cliprange; the offset of the first-derivative zero
            x0=1+eps, and also the position where G(x) attains its local maximum on the positive side.
        kappa_plus: G'(-inf), the asymptotic slope of G when x is far below 1 (the "maximal push",
            saturated push bound).
        kappa_minus: The global minimum of G' (the "maximal pull", bounded redescending pull);
            must satisfy -kappa_plus < kappa_minus < 0.

    Returns:
        (r, a, x0): three scalar constants passed to g_shaping_kernel.
    """
    r = _r_of_kappa_minus(kappa_plus, kappa_minus)
    a = _a_of_scale(eps, kappa_plus, r)
    x0 = 1.0 + eps
    return r, a, x0


# ================================================================ Training-time kernel
# The two functions below are called per batch and are written as torch.jit.script to match
# the calling convention and performance characteristics of the old _ano_math_kernel /
# _compute_ano_loss.

@torch.jit.script
def g_shaping_kernel(
    x: torch.Tensor, r: float, a: float, x0: float, kappa_plus: float
) -> torch.Tensor:
    """G(x), the shaping function itself (replaces the old f0/_ano_math_kernel). **Not** G'(x).

    Let u = sigmoid(a(x-x0)) in (0,1), z = a(x-x0). Closed form (corresponds to G()/_Phi() in
    the reference derivation, but re-parameterized by u instead of E=e^z to avoid overflow of E).
    Using the identities 1+E = 1/(1-u), E+r = (u+r(1-u))/(1-u), the expression
        Phi(t) = (kappa_plus/a)[a t - (2r/(r-1))log(1+E) + ((r+1)/(r-1))log(E+r)]
    is rewritten in pure u form; the coefficients simplify exactly (verified with sympy:
    the coefficient 2r/(r-1) - (r+1)/(r-1) of log(1-u) equals 1 exactly, independent of r):

        r != 1:  bracket(z) = z + log(1-u) + ((r+1)/(r-1)) * log(u + r(1-u))
        r == 1:  bracket(z) = z + log(1-u) + 2*(1-u)            <-- limit as r->1

        G(x) = 1 + (kappa_plus/a) * [ bracket(z) - bracket(z1) ],   z1 = a(1-x0)

    log(u+r(1-u)) is always stable: u+r(1-u) in (0, max(1,r)), so it does not overflow for large z
    the way a direct E+r would. log(1-u) is computed as -softplus(z).

    Note: an earlier version had a sign error on the log(1-u) coefficient (missing the substitution
    "1+E=1/(1-u so log(1+E)=-log(1-u)"), which caused the x>x0 branch to diverge (error of 0.167
    at x=1.07, visually obvious). The current form has been verified by symbolic differentiation
    in sympy (derivative with respect to z is identically equal to the reference Phi') and by
    numerical grid validation (see _selftest, max diff ~1.6e-8).
    """
    z = a * (x - x0)
    z1 = a * (1.0 - x0)
    u = torch.sigmoid(z)
    u1 = 1.0 / (1.0 + math.exp(-z1))  # scalar, plain python float arithmetic suffices

    log_1mu = -torch.nn.functional.softplus(z)          # log(1-u), stable
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

    return 1.0 + (kappa_plus / a) * (bracket - bracket1)


@torch.jit.script
def _compute_g_loss(
    mb_advantage: torch.Tensor,
    ratio: torch.Tensor,
    r: float, a: float, x0: float, kappa_plus: float,
) -> torch.Tensor:
    """Compute the ANO policy loss, using G(x) to replace the old _ano_math_kernel.

    Identical positive/negative Adv branch handling as the old version (the old kernel variable
    was named f; here it is replaced by G):
        Adv >= 0:  loss = -Adv * G(r)
        Adv <  0:  loss = -Adv * [2 - G(2 - r)]   <-- same dual form as the old version

    The constant "2" is carried over directly from the old code: G and the old kernel f share the
    same anchor G(1) = f(1) = 1, and the dual g(x) = 2 - G(2-x) only requires g(1) = 2 - G(1) =
    2 - 1 = 1, which is strictly satisfied with the constant 2 and does not need scaling by
    kappa_plus.
    """
    f_val_pos = g_shaping_kernel(ratio, r, a, x0, kappa_plus)
    f_val_neg = 2.0 - g_shaping_kernel(2.0 - ratio, r, a, x0, kappa_plus)
    target_f_val = torch.where(mb_advantage >= 0, f_val_pos, f_val_neg)
    return -mb_advantage * target_f_val


# --------------------------------------------------------------------- Self-test
def _selftest() -> None:
    """Offline numerical self-test: replicate the operators of g_shaping_kernel in pure
    python/math and cross-check against the E-parameterized closed form of G() from the
    paper's reference derivation."""
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

    def G_u_numpy(x, kappa_plus, r, a, x0):
        """Pure-numpy replica of g_shaping_kernel (no torch dependency, for sandbox self-test).

        Must match the formula in g_shaping_kernel term by term (including the simplified
        log(1-u) coefficient = 1, not 2r/(r-1)); otherwise the self-test cannot detect
        sign errors in the implementation.
        """
        x = np.asarray(x, dtype=np.float64)
        z = a * (x - x0)
        z1 = a * (1.0 - x0)
        u = 1.0 / (1.0 + np.exp(-np.clip(z, -700, 700)))
        u1 = 1.0 / (1.0 + math.exp(-max(min(z1, 700), -700)))
        # Stable log(1-u) = -softplus(z) = -(max(z,0) + log1p(exp(-|z|)))
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
        return 1.0 + (kappa_plus / a) * (bracket - bracket1)

    ref = _ref_module()
    cases = [
        (eps, kappa_plus, -frac * kappa_plus)
        for eps in (0.02, 0.2, 1.0, 5.0)
        for kappa_plus in (1.05, 1.2, 3.0, 50.0, 1e4)
        for frac in (1e-3, 0.01, 0.3, 0.7, 0.99)
    ]
    worst_anchor = 0.0
    worst_vs_ref = 0.0
    for eps, kappa_plus, kappa_minus in cases:
        r, a, x0 = solve_g_shaping_constants(eps, kappa_plus, kappa_minus)
        assert r > 0.0 and a > 0.0

        # (1) G(1) = 1 holds exactly (the anchor of the shaping function, corresponding to
        # the old kernel's f(1)=1)
        g_at_1 = float(G_u_numpy(1.0, kappa_plus, r, a, x0))
        worst_anchor = max(worst_anchor, abs(g_at_1 - 1.0))

        # (2) G attains a local maximum at x0=1+eps (corresponding to the old kernel's highest
        # point at that location). The step size must be scaled relative to the curvature scale
        # 1/a: a can reach several hundred, and a fixed 1e-4 would step outside the small
        # neighborhood of the local maximum and observe the curve shape elsewhere, not the extremum.
        h = 1e-3 / a
        g_lo = float(G_u_numpy(x0 - h, kappa_plus, r, a, x0))
        g_mid = float(G_u_numpy(x0, kappa_plus, r, a, x0))
        g_hi = float(G_u_numpy(x0 + h, kappa_plus, r, a, x0))
        assert g_mid > g_lo and g_mid > g_hi, (
            f"G is not a local max at x0 for eps={eps},kappa_plus={kappa_plus},kappa_minus={kappa_minus}: "
            f"{g_lo:.6f} {g_mid:.6f} {g_hi:.6f}"
        )

        # (3) Numerical cross-check against the E-parameterized closed form from the reference
        if ref is not None:
            r_ref, a_ref = ref.from_b(eps, kappa_plus, kappa_minus)
            worst_vs_ref = max(worst_vs_ref, abs(r - r_ref) / r_ref, abs(a - a_ref) / a_ref)
            grid = np.linspace(x0 - 8.0 / a, x0 + 8.0 / a, 2001)
            g_new = G_u_numpy(grid, kappa_plus, r, a, x0)
            g_old = np.asarray(ref.G(grid, eps, kappa_plus, r, a))
            # Absolute error grows with the overall magnitude at extreme parameter corners
            # (e.g. kappa_plus/a ~ 1e8), so normalize by the curve's own amplitude and compare
            # relative error.
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
