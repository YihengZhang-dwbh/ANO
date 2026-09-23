## ANO Shaping Kernel — Reference Implementation

Pure-numpy reference implementation of the ANO gain-shaping kernel
(paper Section 4 and Appendix G):

- `solve_constants(eps, kappa_plus=10.0, kappa_minus=-1.5)` — closed-form
  internal constants `(nu, a, x0)` from the three knobs `(eps, kappa_plus, kappa_minus)`;
- `gain(x, nu, a, x0, kappa_plus)` — the gain field `G'(x)` (paper Eq. 9);
- `shaping(x, nu, a, x0, kappa_plus)` — the shaping function `G(x)`
  (anchored antiderivative, `G(1) = 1`, `G'(1) = 1`);
- `dual(r, ...)` — the point-symmetric dual `g(r) = 2 - G(2 - r)`;
- `ano_objective(ratio, advantage, eps, kappa_plus, kappa_minus)` — the
  two-sided surrogate `E[min(g(r) A, f(r) A)]` with `f = G`.

Running `python g_shaping.py` executes the construction checks (design
requirements R1–R5 and the two-sided enclosure `f(r) <= r <= g(r)`).

The production trainers live in `../Traditional_RL` (Atari / MuJoCo) and
`../RLHF/experimental/ano` (TRL-based RLHF); they implement the same kernel
in torch (JIT-scripted, numerically stabilized in the sigmoid domain).
