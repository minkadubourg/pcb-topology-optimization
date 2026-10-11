"""Conductor between two electrodes: validation of the adjoint gradient.

Plate [-1, 1]^2 between two full-width electrodes, 1 A injected at the top
electrode, bottom grounded. Optional keep-out zone [-0.1, 0.1] x [-0.4, 0.4].

1. Unconstrained steepest descent on J = 1/2 ∫|∇u|² (linear interpolation),
   starting from a central bar: copper fills the whole plate, which is the
   correct optimum without a material budget (less field everywhere).
2. MMA, minimum equivalent resistance (compliance) with copper <= 30 %, with
   and without the keep-out zone. The resistance of the result is compared
   with the bound of vertical bars, R* = L / (sigma * w), w = copper area / L.

Writes figures/plate_*.png and results/electrode_plate.csv.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pcbtopo import SIMP, ConductionModel, Port
from pcbtopo.geometry import electrode_plate
from pcbtopo.interpolation import element_centroids
from pcbtopo.objectives import Compliance, FieldEnergy, VolumeFraction
from pcbtopo.optimize import Problem, mma, projected_gradient_descent
from pcbtopo.plotting import draw_density

ROOT = Path(__file__).resolve().parents[1]
FIG, RES = ROOT / "figures", ROOT / "results"
KEEPOUT = ((-0.1, -0.4), (0.1, 0.4))
PORTS = [Port("top", current=1.0), Port("bottom", voltage=0.0)]


def main():
    FIG.mkdir(exist_ok=True)
    RES.mkdir(exist_ok=True)
    rows = []

    # 1) unconstrained descent, field-energy objective, as in the original study
    mesh = electrode_plate(keepout=KEEPOUT, maxh=0.04)
    model = ConductionModel(mesh, PORTS, fixed={"keepout": 0.0}, interpolation=SIMP(p=1))
    prob = Problem(model, FieldEnergy(model, "design"))
    c = element_centroids(mesh)[model.design_mask]
    rho0 = (np.abs(c[:, 0]) < 0.5).astype(float)
    snaps = {}
    rho, hist = projected_gradient_descent(prob, rho0, step=0.1, maxiter=200,
                                           callback=lambda k, r: snaps.__setitem__(k - 1, r.copy()))
    rows.append(dict(case="descent, no budget", accepted_steps=len(hist.objective) - 1,
                     J_initial=hist.objective[0], J_final=hist.objective[-1],
                     copper_fraction=float(rho.mean())))
    n = max(snaps)
    picks = [round(n * f) for f in (0.6, 0.8)]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.9))
    panels = [("initial design", rho0), *[(f"after {k} accepted steps", snaps[k]) for k in picks],
              (f"final ({n} steps)", rho)]
    for ax, (title, r) in zip(axes, panels):
        draw_density(ax, model, model.full_density(r), title)
    fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(FIG / "plate_descent.png", dpi=150); plt.close(fig)

    # 2) MMA, minimum resistance with a copper budget
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.9))
    for ax, keepout in zip(axes, (None, KEEPOUT)):
        mesh = electrode_plate(keepout=keepout, maxh=0.03)
        fixed = {"keepout": 0.0} if keepout else {}
        model = ConductionModel(mesh, PORTS, fixed=fixed, interpolation=SIMP(p=3))
        areas = model.element_area[model.design_mask]
        prob = Problem(model, Compliance(model), [VolumeFraction(areas, 0.3)], filter_radius=0.06)
        rho, hist = mma(prob, 0.3 * np.ones(model.n_design), maxiter=200)
        rho_f = prob.filter(rho)
        R = model.solve(model.full_density(rho_f)).port_voltage["top"]          # 1 A injected
        R_bin = model.solve(model.full_density((rho_f > 0.5).astype(float))).port_voltage["top"]
        width = 0.3 * areas.sum() / 2.0                                          # plate height 2
        rows.append(dict(case="MMA, 30 % copper" + (", keep-out" if keepout else ""),
                         iterations=len(hist.objective) - 1, R=R, R_thresholded=R_bin,
                         R_bound=2.0 / width, copper_fraction=float(areas @ rho_f / areas.sum())))
        draw_density(ax, model, model.full_density(rho_f),
                     f"R = {R:.2f} Ω (bound {2.0 / width:.2f} Ω)" + ("\nwith keep-out" if keepout else "\n"))
    fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(FIG / "plate_mma.png", dpi=150); plt.close(fig)

    pd.DataFrame(rows).to_csv(RES / "electrode_plate.csv", index=False, float_format="%.4f")
    print(pd.DataFrame(rows))


if __name__ == "__main__":
    main()
