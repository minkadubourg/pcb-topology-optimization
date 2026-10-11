"""Time-harmonic A–V model of a two-plane board (3D).

Core 1 x 1 x 0.1 (sigma = 1 S/m, eps_r = 1) between two copper planes
(sigma = 1e6 S/m). The top plane is driven at 10 V, the bottom one at -10j V
(90° out of phase); omega = 10 rad/s.

Checks:
* the copper planes are equipotential and V is linear across the core;
* Ampère's law: the circulation of B around a horizontal loop in the core
  equals mu0 times the enclosed current (conduction + displacement).

Writes figures/harmonic_*.png and results/harmonic_board.csv.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pcbtopo.geometry import two_plate_board
from pcbtopo.harmonic import EPS0, MU0, solve_weakly_coupled

ROOT = Path(__file__).resolve().parents[1]
FIG, RES = ROOT / "figures", ROOT / "results"
H, CU, OMEGA, SIGMA_CORE = 0.1, 0.005, 10.0, 1.0
ELECTRODES = {"top": 10.0, "bottom": -10.0j}


def circulation(sol, mesh, half, z, n=400):
    """∮ B·dl around the square |x-0.5|, |y-0.5| = half at height z (counter-clockwise)."""
    s = np.linspace(-half, half, n, endpoint=False) + half / n
    dl = 2 * half / n
    total = 0.0
    for (x0, y0), (dx, dy), (tx, ty) in [((0.5 + half, None), (0, 1), (0, 1)), ((None, 0.5 + half), (-1, 0), (-1, 0)),
                                         ((0.5 - half, None), (0, -1), (0, -1)), ((None, 0.5 - half), (1, 0), (1, 0))]:
        for si in s:
            x = x0 if x0 is not None else 0.5 - dx * si
            y = y0 if y0 is not None else 0.5 + dy * si
            B = np.array(sol.B(mesh(x, y, z)))
            total += (B[0] * tx + B[1] * ty) * dl
    return total


def main():
    FIG.mkdir(exist_ok=True)
    RES.mkdir(exist_ok=True)
    mesh = two_plate_board(thickness=H, copper=CU, maxh=0.05)
    sol = solve_weakly_coupled(mesh, {"core": SIGMA_CORE, "plus": 1e6, "minus": 1e6}, ELECTRODES, OMEGA, order=2)

    # potential across the stack, at three phases of the cycle
    z = np.linspace(-CU, H + CU, 111)
    V = np.array([sol.V(mesh(0.5, 0.5, zi)) for zi in z])
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    for phase, label in [(-np.pi / 2, "−π/2"), (0.0, "0"), (np.pi / 2, "π/2")]:
        ax.plot(np.real(V * np.exp(1j * phase)), z, label=f"phase {label}")
    for zz in (0, H):
        ax.axhline(zz, color="0.6", lw=0.8, ls="--")
    ax.set_xlabel("V (V)"); ax.set_ylabel("z"); ax.legend(frameon=False, fontsize=8)
    ax.set_title("Potential across the board", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "harmonic_potential.png", dpi=150); plt.close(fig)

    V_exact = ELECTRODES["bottom"] + (ELECTRODES["top"] - ELECTRODES["bottom"]) * np.clip(z / H, 0, 1)
    err = np.abs(V - V_exact).max() / abs(ELECTRODES["top"] - ELECTRODES["bottom"])

    # |B| in the mid-plane of the core
    g = np.linspace(0.01, 0.99, 60)
    Bmag = np.array([[np.linalg.norm(np.abs(np.array(sol.B(mesh(x, y, H / 2))))) for x in g] for y in g])
    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    im = ax.imshow(Bmag * 1e6, origin="lower", extent=(0, 1, 0, 1), cmap="viridis")
    fig.colorbar(im, ax=ax, label="|B| (µT)")
    ax.set_title("Magnetic flux density, mid-plane of the core", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "harmonic_B.png", dpi=150); plt.close(fig)

    # Ampère's law
    E_z = -(ELECTRODES["top"] - ELECTRODES["bottom"]) / H
    rows = []
    for half in (0.1, 0.2, 0.3):
        I = (SIGMA_CORE + 1j * OMEGA * EPS0) * E_z * (2 * half) ** 2
        circ = circulation(sol, mesh, half, H / 2)
        rows.append(dict(loop_half_width=half, circulation=abs(circ), mu0_I=abs(MU0 * I),
                         relative_error=abs(circ - MU0 * I) / abs(MU0 * I)))
    df = pd.DataFrame(rows)
    df["max_rel_error_V_linear"] = err
    df.to_csv(RES / "harmonic_board.csv", index=False, float_format="%.4g")
    print(df)


if __name__ == "__main__":
    main()
