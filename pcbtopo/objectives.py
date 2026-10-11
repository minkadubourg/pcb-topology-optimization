"""Objective functions of the state vector x = [interior nodes..., pads...].

Each objective returns its value and its derivative with respect to x; the
adjoint solve in ConductionModel turns the latter into a design gradient.
"""
from __future__ import annotations

import numpy as np


class PortVoltageTarget:
    """J = 1/2 (U_a - U_b - target)^2: drive the voltage across a component."""

    def __init__(self, model, a: str, b: str, target: float):
        self.ia = model.n_interior + model.port_index[a]
        self.ib = model.n_interior + model.port_index[b]
        self.n, self.target = model.n, target

    def __call__(self, x):
        d = x[self.ia] - x[self.ib] - self.target
        g = np.zeros(self.n)
        g[self.ia], g[self.ib] = d, -d
        return 0.5 * d * d, g


class Compliance:
    """J = sum_p I_p U_p: electrical power delivered by the current sources,
    equal to the Joule losses. Minimising it minimises the equivalent resistance."""

    def __init__(self, model):
        self.f = model.f

    def __call__(self, x):
        return float(self.f @ x), self.f.copy()


class FieldEnergy:
    """J = 1/2 ∫_region |∇u|^2 dx: the objective of the original Poisson study."""

    def __init__(self, model, region: str | None = None):
        self.M = model.unit_stiffness(region)

    def __call__(self, x):
        Mx = self.M @ x
        return 0.5 * float(x @ Mx), Mx


class VolumeFraction:
    """g = sum_e A_e rho_e / A_design - limit  (<= 0)."""

    def __init__(self, areas: np.ndarray, limit: float):
        self.w = areas / areas.sum()
        self.limit = limit

    def __call__(self, rho):
        return float(self.w @ rho) - self.limit, self.w.copy()


class Greyness:
    """g = sum_e A_e rho_e (1 - rho_e) / A_design - limit  (<= 0).

    The amount of intermediate material, as in the original study. Note that
    rho (1 - rho) <= 1/4, so a limit of 0.25 or more is never active.
    """

    def __init__(self, areas: np.ndarray, limit: float):
        self.w = areas / areas.sum()
        self.limit = limit

    def __call__(self, rho):
        return float(self.w @ (rho * (1 - rho))) - self.limit, self.w * (1 - 2 * rho)
