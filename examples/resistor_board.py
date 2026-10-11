"""Routing a supply to a resistor: MMA with a copper-area constraint.

Board [-1, 1]^2, supply connector P (10 V), return connector N (0 V), a 10 Ω
resistor between leads Rp and Rn. Copper conductivity 1, empty board 1e-6
(contrast 1e6), copper area <= 30 % of the board.

Objective  J = 1/2 (U_Rp - U_Rn - U_target)^2.

Runs:
  main        SIMP p=3, filter r=0.06, U_target = +10 V and -10 V (reversed polarity)
  variants    interpolation / constraint variants, and a current-driven supply
  contrast    sigma_min = 1e-2, 1e-4, 1e-6

Writes figures/resistor_*.png and results/resistor_board.csv.
"""
from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pcbtopo import SIMP, ConductionModel, Port, Resistor
from pcbtopo.geometry import resistor_board
from pcbtopo.objectives import Greyness, PortVoltageTarget, VolumeFraction
from pcbtopo.optimize import Problem, mma
from pcbtopo.plotting import draw_density, draw_field, element_values

ROOT = Path(__file__).resolve().parents[1]
FIG, RES = ROOT / "figures", ROOT / "results"
VOL, R_LOAD, V_SUPPLY = 0.3, 10.0, 10.0


def build(mesh, interp, supply="voltage"):
    p = Port("P", voltage=V_SUPPLY) if supply == "voltage" else Port("P", current=V_SUPPLY / R_LOAD)
    return ConductionModel(mesh, [p, Port("N", voltage=0.0), Port("Rp"), Port("Rn")],
                           [Resistor("Rp", "Rn", R_LOAD)], interpolation=interp)


def run(mesh, target=V_SUPPLY, interp=None, radius=0.06, grey=None, supply="voltage", maxiter=200):
    model = build(mesh, interp or SIMP(p=3))
    areas = model.element_area[model.design_mask]
    cons = [VolumeFraction(areas, VOL)] + ([Greyness(areas, grey)] if grey is not None else [])
    prob = Problem(model, PortVoltageTarget(model, "Rp", "Rn", target), cons, filter_radius=radius)
    t0 = time.time()
    rho, hist = mma(prob, VOL * np.ones(model.n_design), maxiter=maxiter)
    rho_f = prob.filter(rho)
    state = model.solve(model.full_density(rho_f))
    # performance of the thresholded (manufacturable) layout, same physics
    binary = model.full_density((rho_f > 0.5).astype(float))
    state_bin = model.solve(binary)
    vr = lambda s: s.port_voltage["Rp"] - s.port_voltage["Rn"]
    w = areas / areas.sum()
    return dict(model=model, prob=prob, rho=rho_f, hist=hist, state=state,
                row=dict(target_V=target, iterations=len(hist.objective) - 1,
                         seconds=round(time.time() - t0, 1),
                         V_R=vr(state), V_R_thresholded=vr(state_bin),
                         copper_fraction=float(w @ rho_f),
                         greyness=float(w @ (4 * rho_f * (1 - rho_f)))))


def main():
    FIG.mkdir(exist_ok=True)
    RES.mkdir(exist_ok=True)
    mesh = resistor_board(maxh=0.03)
    rows = []

    # ---- main runs: both polarities
    main_runs = {}
    for target in (+V_SUPPLY, -V_SUPPLY):
        r = run(mesh, target)
        main_runs[target] = r
        rows.append({"run": "SIMP p=3 + filter", **r["row"]})
        print(rows[-1])

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.1))
    for ax, (target, r) in zip(axes[:2], main_runs.items()):
        m = r["model"]
        draw_density(ax, m, m.full_density(r["rho"]),
                     f"target U_R = {target:+.0f} V  →  U_R = {r['row']['V_R']:+.2f} V")
    r = main_runs[+V_SUPPLY]
    jmag = element_values(mesh, r["model"].current_density(r["model"].full_density(r["rho"]), r["state"]))
    im = draw_field(axes[2], mesh, np.where(r["model"].conducting, jmag, np.nan), "|current density| (log10)", log=True,
                    cmap="magma")
    fig.colorbar(im, ax=axes[2], fraction=0.046)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(FIG / "resistor_layouts.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 3.2))
    for target, r in main_runs.items():
        ax.semilogy(r["hist"].objective, label=f"target {target:+.0f} V")
    ax.set_xlabel("MMA iteration"); ax.set_ylabel("J"); ax.legend(frameon=False); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG / "resistor_convergence.png", dpi=150); plt.close(fig)

    # ---- variants
    variants = {
        "linear, no filter": dict(interp=SIMP(p=1), radius=0.0),
        "linear + greyness <= 0.05": dict(interp=SIMP(p=1), radius=0.0, grey=0.05),
        "linear, current-driven supply": dict(interp=SIMP(p=1), radius=0.0, supply="current"),
    }
    var_runs = {}
    for name, kw in variants.items():
        supply = kw.pop("supply", "voltage")
        if supply == "current":
            model = build(mesh, kw["interp"], supply="current")
            areas = model.element_area[model.design_mask]
            prob = Problem(model, PortVoltageTarget(model, "Rp", "Rn", V_SUPPLY), [VolumeFraction(areas, VOL)])
            t0 = time.time()
            rho, hist = mma(prob, VOL * np.ones(model.n_design), maxiter=200)
            st = model.solve(model.full_density(rho))
            stb = model.solve(model.full_density((rho > 0.5).astype(float)))
            w = areas / areas.sum()
            r = dict(model=model, rho=rho, row=dict(
                target_V=V_SUPPLY, iterations=len(hist.objective) - 1, seconds=round(time.time() - t0, 1),
                V_R=st.port_voltage["Rp"] - st.port_voltage["Rn"],
                V_R_thresholded=stb.port_voltage["Rp"] - stb.port_voltage["Rn"],
                copper_fraction=float(w @ rho), greyness=float(w @ (4 * rho * (1 - rho)))))
        else:
            r = run(mesh, V_SUPPLY, **kw)
        var_runs[name] = r
        rows.append({"run": name, **r["row"]})
        print(rows[-1])

    fig, axes = plt.subplots(1, 4, figsize=(15, 4.1))
    panels = [("SIMP p=3 + filter", main_runs[+V_SUPPLY]), *var_runs.items()]
    for ax, (name, r) in zip(axes, panels):
        m = r["model"]
        draw_density(ax, m, m.full_density(r["rho"]),
                     f"{name}\nU_R = {r['row']['V_R']:.2f} V, greyness {r['row']['greyness']:.2f}")
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(FIG / "resistor_variants.png", dpi=150); plt.close(fig)

    # ---- conductivity contrast
    for smin in (1e-2, 1e-4, 1e-6):
        r = run(mesh, V_SUPPLY, interp=SIMP(sigma_min=smin, p=3))
        rows.append({"run": f"SIMP p=3, sigma_min={smin:g}", **r["row"]})
        print(rows[-1])

    pd.DataFrame(rows).to_csv(RES / "resistor_board.csv", index=False, float_format="%.4f")


if __name__ == "__main__":
    main()
