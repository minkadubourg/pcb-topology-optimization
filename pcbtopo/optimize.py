"""Optimisation drivers: projected gradient descent and MMA."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .interpolation import DensityFilter, element_centroids


@dataclass
class History:
    objective: list = field(default_factory=list)
    constraints: list = field(default_factory=list)
    change: list = field(default_factory=list)


class Problem:
    """min_rho J(x(rho))  s.t.  g_i(rho_filtered) <= 0,  0 <= rho <= 1.

    ``rho`` are the design-element densities; the physics sees the filtered
    densities. Gradients are returned with respect to the unfiltered ones.
    """

    def __init__(self, model, objective, constraints=(), filter_radius: float = 0.0):
        self.model, self.objective, self.constraints = model, objective, list(constraints)
        c = element_centroids(model.mesh)[model.design_mask]
        a = model.element_area[model.design_mask]
        self.filter = DensityFilter(c, a, filter_radius)
        self.areas = a

    def evaluate(self, rho):
        m = self.model
        rho_f = self.filter(rho)
        full = m.full_density(rho_f)
        state = m.solve(full)
        J, dJdx = self.objective(state.x)
        dJ = self.filter.backward(m.gradient(full, state, dJdx))
        g, dg = [], []
        for con in self.constraints:
            gi, dgi = con(rho_f)
            g.append(gi)
            dg.append(self.filter.backward(dgi))
        return J, dJ, np.array(g), np.array(dg).reshape(len(g), len(rho)), state


def projected_gradient_descent(problem: Problem, rho0, step=0.1, min_step=1e-4, maxiter=200,
                               grow=1.2, shrink=0.5, callback=None):
    """Steepest descent on the box [0, 1] with step control: a step is accepted
    if it decreases J (step *= grow), otherwise rejected (step *= shrink).
    The gradient is normalised by its maximum so that `step` is a density change.
    This is the algorithm of the original study; it handles no constraint
    other than the bounds.
    """
    rho = np.clip(np.asarray(rho0, float), 0, 1)
    J, dJ, *_ = problem.evaluate(rho)
    hist = History(objective=[J])
    for _ in range(maxiter):
        if step < min_step:
            break
        trial = np.clip(rho - step * dJ / (np.abs(dJ).max() + 1e-300), 0, 1)
        Jt, dJt, *_ = problem.evaluate(trial)
        if Jt < J:
            hist.change.append(np.abs(trial - rho).max())
            rho, J, dJ = trial, Jt, dJt
            hist.objective.append(J)
            step *= grow
            if callback:
                callback(len(hist.objective), rho)
        else:
            step *= shrink
    return rho, hist


def mma(problem: Problem, rho0, maxiter=200, move=0.2, tol=1e-3, callback=None):
    """Method of Moving Asymptotes (Svanberg 1987), via `mmapy` (Deetman).

    Stops when the largest density change between two iterations falls below
    `tol` (after at least 20 iterations) or after `maxiter` iterations.
    """
    from mmapy import mmasub

    x = np.clip(np.asarray(rho0, float), 0, 1).reshape(-1, 1)
    n, m = x.shape[0], len(problem.constraints)
    xmin, xmax = np.zeros((n, 1)), np.ones((n, 1))
    xold1, xold2, low, upp = x.copy(), x.copy(), xmin.copy(), xmax.copy()
    a0, a, c, d = 1.0, np.zeros((m, 1)), 1000 * np.ones((m, 1)), np.ones((m, 1))

    J, dJ, g, dg, _ = problem.evaluate(x.ravel())
    scale = 100.0 / max(abs(J), 1e-30)          # keep J ~ 100 for MMA's default parameters
    hist = History(objective=[J], constraints=[g])
    for it in range(1, maxiter + 1):
        xmma, *_, low, upp = mmasub(m, n, it, x, xmin, xmax, xold1, xold2,
                                     scale * J, scale * dJ.reshape(-1, 1),
                                     g.reshape(-1, 1), dg, low, upp, a0, a, c, d, move)
        xold2, xold1, x = xold1, x, xmma
        J, dJ, g, dg, _ = problem.evaluate(x.ravel())
        change = float(np.abs(x - xold1).max())
        hist.objective.append(J)
        hist.constraints.append(g)
        hist.change.append(change)
        if callback:
            callback(it, x.ravel())
        if it >= 20 and change < tol:
            break
    return x.ravel(), hist
