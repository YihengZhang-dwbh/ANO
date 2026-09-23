"""Generate the README figure: shaping functions and gain fields of
PPO (hard clip), SPO (quadratic penalty), and ANO (this work).

Runs against the reference kernel in ``ano/g_shaping.py``; reproduces with:

    python assets/plot_kernel.py
"""

import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from ano.g_shaping import gain, solve_constants, shaping  # noqa: E402

EPS = 0.2                    # trust-region boundary
KAPPA_PLUS, KAPPA_MINUS = 10.0, -1.5   # paper-recommended knobs

# --- competing shaping functions (paper Section 3.1) -------------------------
def f_ppo(r):
    return np.minimum(r, 1.0 + EPS)

def f_spo(r):
    return -(r - 1.0 - EPS) ** 2 / (2.0 * EPS) + EPS / 2.0 + 1.0

def kappa_ppo(r):
    return np.where(r < 1.0 + EPS, 1.0, 0.0)

def kappa_spo(r):
    return (1.0 + EPS - r) / EPS

# --- ANO from the reference implementation -----------------------------------
nu, a, x0 = solve_constants(EPS, KAPPA_PLUS, KAPPA_MINUS)
f_ano = lambda r: shaping(r, nu, a, x0, KAPPA_PLUS)
kappa_ano = lambda r: gain(r, nu, a, x0, KAPPA_PLUS)

C_PPO, C_SPO, C_ANO, C_ID = "#555555", "#1f77b4", "#d62728", "#bbbbbb"

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))

# Panel (a): shaping functions
r = np.linspace(0.0, 2.3, 1201)
ax = axes[0]
ax.plot(r, r, "--", color=C_ID, lw=1.2, label="identity")
ax.plot(r, f_ppo(r), color=C_PPO, lw=1.8, label="PPO (hard clip)")
ax.plot(r, f_spo(r), color=C_SPO, lw=1.8, label="SPO (quadratic)")
ax.plot(r, f_ano(r), color=C_ANO, lw=2.2, label="ANO (ours)")
ax.plot([1.0], [1.0], "o", color=C_ANO, ms=6)
ax.annotate("anchor (1,1)", xy=(1.0, 1.0), xytext=(1.12, 0.55),
            arrowprops=dict(arrowstyle="->", color=C_ANO), fontsize=9, color=C_ANO)
ax.axvline(1.0 + EPS, color=C_ID, lw=0.9, ls=":")
ax.annotate(r"peak at $1+\epsilon$", xy=(1.0 + EPS, f_ano(1.0 + EPS)),
            xytext=(1.55, 1.75), arrowprops=dict(arrowstyle="->", color=C_ID),
            fontsize=9, color="#666666")
ax.set_xlabel(r"probability ratio $r$")
ax.set_ylabel(r"shaping function $f(r)$")
ax.set_title("(a) Shaping functions", fontsize=11)
ax.set_xlim(0, 2.3)
ax.set_ylim(-6.5, 2.3)
ax.legend(loc="lower right", fontsize=9)
ax.grid(alpha=0.25)

# Panel (b): gain fields
r = np.linspace(0.0, 3.0, 1201)
ax = axes[1]
ax.plot(r, kappa_ppo(r), color=C_PPO, lw=1.8, label="PPO (dead zone)")
ax.plot(r, kappa_spo(r), color=C_SPO, lw=1.8, label="SPO (unbounded)")
ax.plot(r, kappa_ano(r), color=C_ANO, lw=2.2, label="ANO (bounded, redescending)")
ax.axhline(KAPPA_PLUS, color=C_ANO, lw=0.9, ls="--")
ax.axhline(KAPPA_MINUS, color=C_ANO, lw=0.9, ls="--")
ax.annotate(r"saturated push $\kappa_+$", xy=(2.7, KAPPA_PLUS), xytext=(1.75, 7.4),
            arrowprops=dict(arrowstyle="->", color=C_ANO), fontsize=9, color=C_ANO)
ax.annotate(r"pull floor $\kappa_-$", xy=(2.1, KAPPA_MINUS), xytext=(2.15, -4.2),
            arrowprops=dict(arrowstyle="->", color=C_ANO), fontsize=9, color=C_ANO)
ax.axvline(1.0 + EPS, color=C_ID, lw=0.9, ls=":")
ax.annotate(r"stable fixed point $1+\epsilon$", xy=(1.0 + EPS, 0.0), xytext=(1.35, 2.6),
            arrowprops=dict(arrowstyle="->", color=C_ID), fontsize=9, color="#666666")
ax.set_xlabel(r"probability ratio $r$")
ax.set_ylabel(r"gain field $\kappa(r) = f'(r)$")
ax.set_title("(b) Gain fields", fontsize=11)
ax.set_xlim(0, 3.0)
ax.set_ylim(-10.8, 10.8)
ax.legend(loc="lower left", fontsize=9)
ax.grid(alpha=0.25)

fig.tight_layout()
out = pathlib.Path(__file__).resolve().parent / "shaping_kernel.png"
fig.savefig(out, dpi=160)
print(f"written: {out}")
