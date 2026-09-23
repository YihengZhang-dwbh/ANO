"""ANO shaping kernel — numpy reference implementation.

Paper: "ANO: Robust Policy Optimization via Bounded, Redescending Gain Fields".

The surrogate objective is treated as a feedback law on the probability ratio
r = pi_new(a|s) / pi_old(a|s); the derivative of the shaping function G is the
gain field

    G'(x) = kappa_plus * nu * (1 - E) / ((1 + E) * (E + nu)),
    E = exp(a * (x - x0)),   x0 = 1 + eps                (paper Eq. 9)

Design requirements (paper Section 4.1, Table 1):
  R1 anchoring     G(1) = 1, G'(1) = 1
  R2 sign reversal G'(1 + eps) = 0
  R3 asymptotics   G'(-inf) = kappa_plus, G'(+inf) = 0
  R4 extremum      min G' = kappa_minus  (bounded redescending pull)
  R5 parsimony     G'' vanishes and changes sign exactly once

The three knobs (eps, kappa_plus, kappa_minus) have decoupled roles; all
internal constants are recovered in closed form before training:
  - nu     from the normalized depth d = -kappa_minus / kappa_plus
           (paper Eq. 13 and its inversion, Appendix G.2)
  - a      from the anchoring condition G'(1) = 1, a quadratic in q = e^{-a*eps}
           evaluated in the cancellation-free conjugate form (paper Eq. 15)
  - G(x)   antiderivative Phi (paper Eq. 14 / Appendix G.3, with the nu = 1
           limit handled separately)
"""

import math

import numpy as np

__all__ = ["solve_constants", "gain", "shaping", "dual", "ano_objective"]


def _nu_of_depth(kappa_plus: float, kappa_minus: float) -> float:
    """Closed-form shape parameter nu from (kappa_plus, kappa_minus).

    d = -kappa_minus / kappa_plus is the normalized depth; nu is recovered from
    d(nu) = (s^2 - 2) / (s + 2)^2 with s = sqrt(2 + 2*nu) (paper Eq. 13).
    Inversion (Appendix G.2): s = (2d + sqrt(2(d + 1))) / (1 - d).
    """
    if not (kappa_plus > 1.0):
        raise ValueError(f"kappa_plus must be > 1, got {kappa_plus}")
    if not (-kappa_plus < kappa_minus < 0.0):
        raise ValueError(
            f"kappa_minus must satisfy -kappa_plus < kappa_minus < 0 "
            f"(got kappa_minus={kappa_minus}, kappa_plus={kappa_plus})"
        )
    d = -kappa_minus / kappa_plus  # in (0, 1), bijective in nu
    s = (2.0 * d + math.sqrt(2.0 * (d + 1.0))) / (1.0 - d)
    return 0.5 * (s * s - 2.0)


def _a_of_scale(eps: float, kappa_plus: float, nu: float) -> float:
    """Closed-form logistic scale a from the anchoring condition G'(1) = 1.

    With q = E(1) = e^{-a*eps}, the condition is q^2 + B q - C = 0 with
    B = nu*(kappa_plus + 1) + 1 and C = nu*(kappa_plus - 1) (paper Eq. 14).
    The positive root is evaluated in the conjugate form
    q = 2C / (B + sqrt(B^2 + 4C)) to avoid catastrophic cancellation
    (paper Eq. 15); q is guaranteed to lie in (0, 1).
    """
    if not (eps > 0.0):
        raise ValueError(f"eps must be > 0, got {eps}")
    B = nu * (kappa_plus + 1.0) + 1.0
    C = nu * (kappa_plus - 1.0)
    q = 2.0 * C / (B + math.sqrt(B * B + 4.0 * C))
    return -math.log(q) / eps


def solve_constants(eps: float, kappa_plus: float = 10.0, kappa_minus: float = -1.5):
    """Solve all internal constants from the three knobs (closed form).

    Returns (nu, a, x0) with x0 = 1 + eps; gain() and shaping() take these
    together with kappa_plus. Paper defaults: kappa_plus = 10, kappa_minus = -1.5.
    """
    nu = _nu_of_depth(kappa_plus, kappa_minus)
    a = _a_of_scale(eps, kappa_plus, nu)
    return nu, a, 1.0 + eps


def gain(x, nu: float, a: float, x0: float, kappa_plus: float):
    """Gain field G'(x) (paper Eq. 9). Accepts scalars or numpy arrays.

    Evaluated in the overflow-free sigmoid parameterization u = sigmoid(z),
    z = a(x - x0):  G' = kappa_plus * nu * (1 - 2u)(1 - u) / (u + nu * (1 - u)),
    which is algebraically identical to Eq. 9 but finite for all x.
    """
    x = np.asarray(x, dtype=np.float64)
    with np.errstate(over="ignore"):  # exp(-z) overflows only where sigmoid saturates to 0/1
        u = 1.0 / (1.0 + np.exp(-a * (x - x0)))
    return kappa_plus * nu * (1.0 - 2.0 * u) * (1.0 - u) / (u + nu * (1.0 - u))


def _phi(x, nu: float, a: float, x0: float, kappa_plus: float):
    """Antiderivative of the gain field (paper Eq. 14 / Appendix G.3).

    Uses numerically stable forms: log(1 + E) = softplus(z) and
    log(E + nu) = z + log(1 + nu * e^{-z}), with z = a(x - x0).
    """
    x = np.asarray(x, dtype=np.float64)
    z = a * (x - x0)
    if abs(nu - 1.0) < 1e-12:
        # nu = 1 limit: Phi = (kappa_plus / a) * (z - log(1 + E) + 2 / (1 + E)),
        # with 1 / (1 + E) = 1 - sigmoid(z) to stay finite for large z.
        return (kappa_plus / a) * (z - np.logaddexp(0.0, z) + 2.0 * (1.0 - 1.0 / (1.0 + np.exp(-z))))
    c = (nu + 1.0) / (nu - 1.0)
    log1p_E = np.logaddexp(0.0, z)          # log(1 + E)
    log_E_nu = z + np.logaddexp(0.0, math.log(nu) - z)  # log(E + nu)
    return (kappa_plus / a) * (z - (c + 1.0) * log1p_E + c * log_E_nu)


def shaping(x, nu: float, a: float, x0: float, kappa_plus: float):
    """Shaping function G(x) = Phi(x) + 1 - Phi(1) (paper Section 4.2).

    Satisfies G(1) = 1 and G'(1) = 1 exactly by construction. The additive
    constant 1 - Phi(1) anchors the antiderivative at the identity map.
    """
    return _phi(x, nu, a, x0, kappa_plus) + 1.0 - _phi(1.0, nu, a, x0, kappa_plus)


def dual(r, nu: float, a: float, x0: float, kappa_plus: float):
    """Point-symmetric dual g(r) = 2 - G(2 - r); encloses g(r) >= r."""
    return 2.0 - shaping(2.0 - np.asarray(r, dtype=np.float64), nu, a, x0, kappa_plus)


def ano_objective(ratio, advantage, eps: float, kappa_plus: float = 10.0, kappa_minus: float = -1.5):
    """Generalized surrogate E[min(g(r) A, f(r) A)] with f = G, g = 2 - G(2 - .).

    Exact instance of the paper's two-sided enclosure (Definition 3.2);
    recovers PPO-clip exactly for the degenerate interval eps_u = eps_l = 0.
    """
    nu, a, x0 = solve_constants(eps, kappa_plus, kappa_minus)
    r = np.asarray(ratio, dtype=np.float64)
    A = np.asarray(advantage, dtype=np.float64)
    f_r = shaping(r, nu, a, x0, kappa_plus)
    g_r = dual(r, nu, a, x0, kappa_plus)
    return np.minimum(g_r * A, f_r * A)


if __name__ == "__main__":
    # Consistency checks against the construction (R1-R5 and the enclosure).
    for eps in (0.1, 0.2, 0.3):
        for kp, km in ((10.0, -1.5), (2.0, -0.5), (1.5, -1.2)):
            nu, a, x0 = solve_constants(eps, kp, km)
            h = 1e-5
            g1 = (shaping(1.0 + h, nu, a, x0, kp) - shaping(1.0 - h, nu, a, x0, kp)) / (2 * h)
            assert abs(shaping(1.0, nu, a, x0, kp) - 1.0) < 1e-9          # R1
            assert abs(g1 - 1.0) < 1e-6                                    # R1
            assert abs(gain(x0, nu, a, x0, kp)) < 1e-12                    # R2
            assert abs(gain(-50.0, nu, a, x0, kp) - kp) < 1e-6             # R3 left
            assert abs(gain(50.0, nu, a, x0, kp)) < 1e-9                   # R3 right
            grid = np.linspace(-30, 30, 200001)
            assert abs(gain(grid, nu, a, x0, kp).min() - km) < 1e-4        # R4
            assert np.all(shaping(grid, nu, a, x0, kp) <= grid + 1e-9)     # G(x) <= x
            assert np.all(dual(grid, nu, a, x0, kp) >= grid - 1e-9)        # g(r) >= r
    print("all construction checks passed")
