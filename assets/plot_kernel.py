"""Generate the README figure: geometry and induced dynamics of the surrogate
objectives (PPO hard clip, SPO quadratic penalty, ANO), matching Figure 1 of
the paper.

The kernels are imported from the shipped reference implementation
(``ano/g_shaping.py``); the two baseline shaping functions are one-liners.
Every panel uses EQUAL x/y scaling (axes boxes sized manually so the data
aspect matches exactly): slopes are visually honest -- the identity is a true
45-degree reference and ANO is visibly tangent to it at r=1 in panel (a).

Display config: (eps, kappa_plus, kappa_minus) = (0.2, 2, -0.5) -- the same
display config as the paper's Figure 1 (a real grid point), chosen over the
paper-recommended (0.2, 10, -1.5) because the gentler slopes keep the GLOBAL
shape of all three objectives inside one window.

  (a) Shaping functions, r in [0, 2.42], y in [-1.7, 1.5]: ANO's full S-shape.
  (b) Gain fields, r in [0, 4.2], y in [-3.8, 2.7]: bounded vs unbounded.
  (c) Ratio-space phase lines (A > 0): arrow direction = sign of the induced
      velocity, length proportional to |v(r)| (per-row normalized).

Reproduce with:

    python assets/plot_kernel.py
"""

import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from ano.g_shaping import gain, shaping, solve_constants  # noqa: E402

# ---------------- display config (paper Figure 1 display config) --------------
EPS = 0.2
KAPPA_PLUS, KAPPA_MINUS = 2.0, -0.5
NU, A_, X0 = solve_constants(EPS, KAPPA_PLUS, KAPPA_MINUS)

# ---------------- kernels (ANO from the shipped reference module) -------------
def ano_Gp(x):
    """G'(x), the gain field of the ANO shaping function."""
    return gain(x, NU, A_, X0, KAPPA_PLUS)

def ano_G(x):
    """G(x) with anchor G(1) = 1."""
    return shaping(x, NU, A_, X0, KAPPA_PLUS)

def ano_G_inf():
    """Horizontal asymptote G(+inf) (redescending tail); x=40 is far enough
    that G' ~ 0 and small enough that exp() stays exact."""
    return shaping(40.0, NU, A_, X0, KAPPA_PLUS)

# ---------------- baselines ---------------------------------------------------
def ppo_f(x, eps):
    return np.minimum(x, 1.0 + eps)

def ppo_fp(x, eps):
    return (x < 1.0 + eps).astype(float)

def spo_f(x, eps):
    return -(x - 1.0 - eps) ** 2 / (2.0 * eps) + eps / 2.0 + 1.0

def spo_fp(x, eps):
    return (1.0 + eps - x) / eps

# ---------------- layout ------------------------------------------------------
G_INF = ano_G_inf()
G0 = ano_G(0.0)
XP = X0 + np.log1p(np.sqrt(2 + 2 * NU)) / A_            # inflection point x_p
GP_XP = gain(XP, NU, A_, X0, KAPPA_PLUS)                # = kappa_minus
XMAX = 2.42              # (a): SPO exits the bottom edge here
XMAX_BC = 4.2            # (b): full tail visible
XMAX_C = 3.4             # (c): ANO arrows vanish by r~2, SPO saturates by r~2.2

C_PPO, C_SPO, C_ANO = "#6b7a8d", "#e8973d", "#c1272d"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 7.5,
    "axes.linewidth": 0.8,
    "mathtext.fontset": "stix",
})

# manual geometry: every axes box matches its data aspect exactly (equal scale)
HBOX = 2.20                                            # axes height, inches
W_A = HBOX * XMAX / 3.2                                # (a) 2.42 x 3.2
W_B = HBOX * XMAX_BC / 6.5                             # (b) 4.2 x 6.5
W_C = HBOX * XMAX_C / 3.4                              # (c) 3.4 x 3.4
ML, GAP, YLAB, MR, MB, MT = 0.40, 0.30, 0.33, 0.06, 0.30, 0.20
FIGW = ML + W_A + GAP + YLAB + W_B + GAP + W_C + MR
FIGH = MB + HBOX + MT
fig = plt.figure(figsize=(FIGW, FIGH))
axes = [fig.add_axes([ML / FIGW, MB / FIGH, W_A / FIGW, HBOX / FIGH]),
        fig.add_axes([(ML + W_A + GAP + YLAB) / FIGW, MB / FIGH, W_B / FIGW, HBOX / FIGH]),
        fig.add_axes([(ML + W_A + GAP + YLAB + W_B + GAP) / FIGW, MB / FIGH,
                      W_C / FIGW, HBOX / FIGH])]

# ---- (a) shaping functions: EQUAL aspect, compact window --------------------
ax = axes[0]
ax.set_aspect("equal")
x = np.linspace(0.0, XMAX, 3000)
ax.plot(x, x, "k--", lw=0.9, alpha=0.5, label=r"identity $y=x$")
ax.plot(x, ppo_f(x, EPS), color=C_PPO, lw=1.5, label=r"PPO (hard clip)")
ax.plot(x, spo_f(x, EPS), color=C_SPO, lw=1.5, label=r"SPO (quadratic)")
ax.plot(x, ano_G(x), color=C_ANO, lw=1.8, label=r"ANO (ours)")
# ANO right-tail asymptote (slope -> 0), drawn dashed
ax.plot([1.65, XMAX], [G_INF, G_INF], color=C_ANO, lw=0.8, ls=(0, (4, 3)), alpha=0.7)
ax.annotate(r"tail slope $\to 0$", xy=(2.15, 0.80),
            xytext=(2.02, 0.28), fontsize=6.8, color=C_ANO, ha="center",
            arrowprops=dict(arrowstyle="->", color=C_ANO, lw=0.7))
# ANO left tail: bounded slope kappa_plus
ax.annotate(r"slope $\to \kappa_{+}{=}2$ (bounded push)",
            xy=(0.12, ano_G(0.12) - 0.05), xytext=(0.92, -1.28),
            fontsize=6.8, color=C_ANO, ha="center",
            arrowprops=dict(arrowstyle="->", color=C_ANO, lw=0.7))
# tangency + peak markers
ax.plot([1.0], [1.0], marker="o", ms=3.6, color="k", mec="k")
ax.annotate("anchor $G(1){=}1$,\n$G'(1){=}1$", xy=(0.96, 0.96), xytext=(0.04, 0.98),
            fontsize=6.8, ha="left", arrowprops=dict(arrowstyle="->", lw=0.6, color="0.25"))
G_PEAK = ano_G(X0)
ax.plot([X0], [G_PEAK], marker="o", ms=3.6, mfc="none", mec=C_ANO, mew=1.1)
# SPO enters from below the left edge and exits through the bottom (unbounded)
ax.annotate("SPO exits frame (unbounded)", xy=(2.40, -1.66),
            xytext=(1.35, -1.52), fontsize=6.8, color=C_SPO, ha="center",
            arrowprops=dict(arrowstyle="->", color=C_SPO, lw=0.7))
ax.axvline(1.0, color="k", lw=0.6, ls=":", alpha=0.5)
ax.axvline(1.0 + EPS, color="k", lw=0.6, ls=":", alpha=0.5)
ax.annotate(r"$r=1$", xy=(0.97, -0.28), ha="right", fontsize=6.8)
ax.annotate(r"$r=1+\epsilon$", xy=(1.25, -0.58), ha="left", fontsize=6.8)
ax.set_xlim(0, XMAX)
ax.set_ylim(-1.7, 1.5)
ax.set_xlabel(r"probability ratio $r$", fontsize=7.5)
ax.set_ylabel(r"shaping function $f(r)$", fontsize=7.5)
ax.set_title("(a) Shaping functions", fontsize=7.8, pad=1.5)
ax.tick_params(labelsize=6.8)
ax.legend(fontsize=6.2, loc="upper right", frameon=False, framealpha=0.9,
          handlelength=1.4, borderaxespad=0.1)

# ---- (b) gain fields: bounded vs unbounded ----------------------------------
ax = axes[1]
ax.set_aspect("equal")
x = np.linspace(0.0, XMAX_BC, 4000)
ax.axhline(0, color="k", lw=0.7, alpha=0.5)
# PPO: draw the actual discontinuity (drop at the boundary)
ax.plot([0, 1.0 + EPS], [1, 1], color=C_PPO, lw=1.5, label="PPO")
ax.plot([1.0 + EPS, 1.0 + EPS], [1, 0], color=C_PPO, lw=1.5)
ax.plot(x[x >= 1.0 + EPS], ppo_fp(x[x >= 1.0 + EPS], EPS), color=C_PPO, lw=1.5)
ax.plot(x, spo_fp(x, EPS), color=C_SPO, lw=1.5, label="SPO")
ax.plot(x, ano_Gp(x), color=C_ANO, lw=1.8, label="ANO (ours)")
# saturated push asymptote kappa_plus (ANO approaches it as r -> 0)
ax.plot([0.0, 0.6], [KAPPA_PLUS, KAPPA_PLUS], color=C_ANO, lw=0.8, ls=(0, (4, 3)), alpha=0.75)
ax.annotate(r"saturated push $\kappa_{+}{=}2$", xy=(1.38, 2.30), fontsize=6.8, color=C_ANO,
            ha="center")
# pull minimum marker (x_p, kappa_minus)
ax.plot([XP], [GP_XP], marker="o", ms=3.6, mfc="none", mec=C_ANO, mew=1.2)
ax.annotate(r"max pull $\kappa_{-}{=}-0.5$" "\n" r"at $r{=}x_p$", xy=(XP + 0.05, GP_XP - 0.14),
            xytext=(2.30, -1.38), fontsize=6.8, color=C_ANO, ha="center",
            arrowprops=dict(arrowstyle="->", color=C_ANO, lw=0.7))
# redescending tail annotation
ax.annotate(r"redescends to $0$", xy=(3.80, -0.03),
            xytext=(3.55, 1.35), fontsize=6.8, color=C_ANO, ha="center",
            arrowprops=dict(arrowstyle="->", color=C_ANO, lw=0.7))
# dead-zone shading for PPO
ax.axvspan(1.0 + EPS, XMAX_BC, color=C_PPO, alpha=0.07)
ax.annotate("PPO dead zone:\nzero feedback", xy=(2.75, 0.55), fontsize=6.8,
            color=C_PPO, ha="center")
ax.annotate("SPO unbounded:\nexits the frame", xy=(1.96, -3.72),
            xytext=(3.02, -2.58), fontsize=6.8, color=C_SPO, ha="center",
            arrowprops=dict(arrowstyle="->", color=C_SPO, lw=0.7))
ax.axvline(1.0, color="k", lw=0.6, ls=":", alpha=0.5)
ax.axvline(1.0 + EPS, color="k", lw=0.6, ls=":", alpha=0.5)
ax.set_xlim(0, XMAX_BC)
ax.set_ylim(-3.8, 2.7)
ax.set_xlabel(r"probability ratio $r$", fontsize=7.5)
ax.set_ylabel(r"gain field $f'(r)$", fontsize=7.5)
ax.set_title("(b) Induced feedback gain", fontsize=7.8, pad=1.5)
ax.tick_params(labelsize=6.8)
ax.legend(fontsize=6.2, loc="lower left", frameon=True, framealpha=0.92,
          edgecolor="none", facecolor="white", handlelength=1.4,
          borderaxespad=0.1)

# ---- (c) ratio-space phase lines (A > 0) ------------------------------------
ax = axes[2]
ax.set_aspect("equal")
rows = [("PPO", C_PPO, 2.0, lambda t: ppo_fp(t, EPS)),
        ("SPO", C_SPO, 1.0, lambda t: spo_fp(t, EPS)),
        ("ANO", C_ANO, 0.0, ano_Gp)]
R0, R1 = 0.25, XMAX_C - 0.1
DELTA = 0.17          # max arrow half-length in r units (per-row normalized);
                      # 2*DELTA = 0.34 < grid spacing 0.42, so arrows never fuse
BOUND = 1.0 + EPS     # trust-region boundary: PPO force stops here
# strictly uniform r grid, identical on every row: cross-row length comparison.
# grid starts right of the row labels so arrows never touch "PPO"/"SPO"/"ANO"
ARROW_XS = [0.55, 0.97, 1.39, 1.81, 2.23, 2.65, 3.07]
for name, color, y, vfun in rows:
    xs = np.linspace(R0, R1, 6000)
    vs = vfun(xs)
    vmax = np.abs(vs).max()
    ax.axhline(y, color="0.75", lw=1.0, zorder=1)
    ax.annotate(name, xy=(0.05, y), fontsize=7.2, color=color, va="center",
                ha="left", fontweight="bold")
    # skip near-zero velocity (no force there); PPO arrows are clipped so that
    # NONE of them crosses into the dead zone
    for x0_ in ARROW_XS:
        v = vfun(np.array([x0_]))[0]
        if abs(v) < 0.03 * vmax:
            continue
        dx = DELTA * v / vmax
        x_start, x_end = x0_ - dx, x0_ + dx
        if x_end > BOUND and v > 0 and name == "PPO":
            x_end = BOUND - 0.02
            if x_end <= x_start:
                continue
        ax.annotate("", xy=(x_end, y), xytext=(x_start, y),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.1,
                                    alpha=0.9, mutation_scale=7.5), zorder=3)
# PPO dead-zone highlight
ax.axvspan(BOUND, R1, ymin=0.735, ymax=0.985, color=C_PPO, alpha=0.08)
ax.annotate("no feedback:\nopen-loop drift", xy=(2.45, 2.26), fontsize=6.8,
            color=C_PPO, ha="center", va="center")
# fixed points for SPO & ANO at r = 1 + eps
for _, color, y, _ in rows[1:]:
    ax.plot([BOUND], [y], marker="o", ms=6, mfc="white", mec=color, mew=1.4,
            zorder=4)
ax.annotate("stable fixed point $r{=}1{+}\\epsilon$" "\n"
            r"(SPO: stiff, ANO: gentle, bounded)",
            xy=(1.29, 0.08), xytext=(2.32, 0.60), fontsize=6.8, ha="center",
            arrowprops=dict(arrowstyle="->", lw=0.8, color="0.2"))
ax.annotate("ANO force redescends:\noutliers barely pushed", xy=(2.25, -0.14),
            xytext=(1.55, -0.60), fontsize=6.8, color=C_ANO, ha="center",
            arrowprops=dict(arrowstyle="->", color=C_ANO, lw=0.7))
ax.annotate("SPO force stays strong\nfar from equilibrium", xy=(2.85, 1.10),
            xytext=(2.30, 1.46), fontsize=6.8, color=C_SPO, ha="center",
            arrowprops=dict(arrowstyle="->", color=C_SPO, lw=0.7))
ax.axvline(1.0, color="k", lw=0.6, ls=":", alpha=0.5)
ax.axvline(BOUND, color="k", lw=0.6, ls=":", alpha=0.5)
ax.set_xlim(0, XMAX_C)
ax.set_ylim(-0.85, 2.55)
ax.set_yticks([])
ax.set_xlabel(r"ratio $r$  (uniform grid; length $\propto|v(r)|$, per-row norm.)",
              fontsize=7.5)
ax.set_title(r"(c) Phase lines ($A>0$)", fontsize=7.8, pad=1.5)
ax.tick_params(labelsize=6.8)

out = pathlib.Path(__file__).resolve().parent / "shaping_kernel.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"written: {out}")

# sanity checks on the shipped kernel (run every time the figure regenerates)
print(f"nu={NU:.6f} a={A_:.6f} x0={X0:.3f}")
print(f"G(1)={ano_G(1.0):.6f}  G'(1)={ano_Gp(1.0):.6f}")
print(f"G'(1+eps)={ano_Gp(1.0 + EPS):.2e}")
print(f"G(0)={G0:.4f}  G(inf)={G_INF:.4f}  x_p={XP:.4f}  G'(x_p)={GP_XP:.4f}")
xx = np.linspace(-5, 8, 200001)
gg = ano_Gp(xx)
print(f"min G' = {gg.min():.6f} (target {KAPPA_MINUS}), "
      f"G'(-5)={gg[0]:.4f} (target {KAPPA_PLUS}),  G'(8)={gg[-1]:.2e}")
