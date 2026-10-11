r"""Steady conduction on a board with pads and lumped components.

Model (2D, per unit thickness of copper)
----------------------------------------
The electric potential :math:`u` satisfies :math:`\nabla\cdot(\sigma(\rho)\nabla u) = 0`
in the conducting region :math:`\Omega` (design region plus any region with a
prescribed density). Each pad :math:`p` is an equipotential contact with an
unknown voltage :math:`U_p`. The pads are coupled by

* injected currents :math:`I_p` (current sources at connectors), and
* lumped two-terminal components, here linear resistors :math:`R_c`
  between pads :math:`a_c` and :math:`b_c`.

Weak form: find :math:`u = u_0 + \sum_p U_p\,\phi_p` with :math:`u_0 = 0` on the pad
boundaries and :math:`\phi_p` a nodal function equal to 1 on pad :math:`p`, such that
for every test pair :math:`(v_0, V_p)`

.. math::
    \int_\Omega \sigma(\rho)\,\nabla u\cdot\nabla v\,dx
    + \sum_c \frac{1}{R_c}(U_{a_c}-U_{b_c})(V_{a_c}-V_{b_c}) = \sum_p I_p V_p .

The pad voltages are the "port" unknowns; the component term is the current
balance at each pad (Kirchhoff's current law). For a linear resistor this is
equivalent to the Lagrange-multiplier formulation with the component current
as an extra unknown, but it keeps the system symmetric, so the adjoint problem
reuses the factorisation of the direct one.

Discretisation: P1 Lagrange elements for :math:`u_0`, one scalar per pad, one
design variable per element. Unknowns are ordered ``[interior nodes..., pads...]``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu
from ngsolve import (H1, L2, VOL, BilinearForm, CoefficientFunction, GridFunction,
                     Integrate, dx, grad)

from .interpolation import SIMP


@dataclass
class Port:
    """A pad. Either its current is imposed (``current``, default 0: floating)
    or its voltage is imposed (``voltage``; ground is ``voltage=0``)."""
    name: str
    current: float = 0.0
    voltage: float | None = None


@dataclass
class Resistor:
    a: str
    b: str
    resistance: float


@dataclass
class State:
    x: np.ndarray                  # reduced vector [interior nodes, pads]
    u: GridFunction                # full potential field (pads excluded)
    port_voltage: dict = field(default_factory=dict)


class ConductionModel:
    """Finite-element conduction model with one density per element.

    Parameters
    ----------
    mesh : ngsolve.Mesh
    ports : list of Port
        Pads (materials) with their boundary named ``pad_<name>``.
    resistors : list of Resistor
    design : str
        Material name of the design region.
    fixed : dict
        Prescribed densities of other conducting materials, e.g. ``{"keepout": 0}``.
    interpolation : SIMP
        sigma(rho) law.
    """

    def __init__(self, mesh, ports, resistors=(), design="design", fixed=None,
                 interpolation: SIMP | None = None):
        self.mesh = mesh
        self.ports = list(ports)
        self.resistors = list(resistors)
        self.interp = interpolation or SIMP()
        fixed = dict(fixed or {})
        self.port_index = {p.name: i for i, p in enumerate(self.ports)}

        regions = [design, *fixed]
        self.region = "|".join(regions)
        mats = np.array([el.mat for el in mesh.Elements(VOL)])
        self.design_mask = mats == design
        self.conducting = np.isin(mats, regions)
        self.rho_fixed = np.zeros(mesh.ne)
        for name, value in fixed.items():
            self.rho_fixed[mats == name] = value
        self.element_area = np.array(Integrate(CoefficientFunction(1), mesh, element_wise=True))
        self.n_design = int(self.design_mask.sum())

        pad_bcs = "|".join(f"pad_{p.name}" for p in self.ports)
        self.fes = H1(mesh, order=1, definedon=mesh.Materials(self.region), dirichlet=pad_bcs)
        free = np.array([self.fes.FreeDofs()[i] for i in range(self.fes.ndof)])
        self.interior = np.flatnonzero(free)
        n_int, n_port = len(self.interior), len(self.ports)

        # T maps [interior..., pads...] to the nodal vector of u
        rows, cols = list(self.interior), list(range(n_int))
        for k, p in enumerate(self.ports):
            on_pad = self.fes.GetDofs(mesh.Boundaries(f"pad_{p.name}"))
            dofs = [i for i in range(self.fes.ndof) if on_pad[i]]
            if not dofs:
                raise ValueError(f"no boundary named pad_{p.name}")
            rows += dofs
            cols += [n_int + k] * len(dofs)
        self.T = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(self.fes.ndof, n_int + n_port))
        self.n = n_int + n_port
        self.n_interior = n_int

        L = np.zeros((n_port, n_port))
        for r in self.resistors:
            a, b, g = self.port_index[r.a], self.port_index[r.b], 1.0 / r.resistance
            L[a, a] += g; L[b, b] += g; L[a, b] -= g; L[b, a] -= g
        self.L = sp.block_diag([sp.csr_matrix((n_int, n_int)), sp.csr_matrix(L)]).tocsr()

        self.f = np.zeros(self.n)
        self.prescribed = np.zeros(self.n, bool)
        self.x_prescribed = np.zeros(self.n)
        for k, p in enumerate(self.ports):
            if p.voltage is None:
                self.f[n_int + k] = p.current
            else:
                self.prescribed[n_int + k] = True
                self.x_prescribed[n_int + k] = p.voltage
        if not self.prescribed.any():
            raise ValueError("at least one port needs an imposed voltage (ground)")
        self.unknown = ~self.prescribed

        self._sigma = GridFunction(L2(mesh, order=0))
        u, v = self.fes.TnT()
        self._stiffness = BilinearForm(self.fes)
        self._stiffness += self._sigma * grad(u) * grad(v) * dx(definedon=mesh.Materials(self.region))
        self._lu = None

    # ------------------------------------------------------------------ densities
    def full_density(self, rho_design: np.ndarray) -> np.ndarray:
        """Element densities on the whole mesh from the design-element densities."""
        rho = self.rho_fixed.copy()
        rho[self.design_mask] = rho_design
        return rho

    def stiffness_matrix(self, rho_full: np.ndarray, sigma=None) -> sp.csr_matrix:
        self._sigma.vec.FV().NumPy()[:] = self.interp(rho_full) if sigma is None else sigma
        self._stiffness.Assemble()
        r, c, v = self._stiffness.mat.COO()
        K = sp.csr_matrix((np.array(v), (np.array(r), np.array(c))), shape=(self.fes.ndof,) * 2)
        return (self.T.T @ K @ self.T + self.L).tocsr()

    def unit_stiffness(self, region: str | None = None) -> sp.csr_matrix:
        """Reduced matrix of  ∫_region ∇u·∇v  (sigma = 1), for field-energy objectives."""
        u, v = self.fes.TnT()
        a = BilinearForm(self.fes)
        a += grad(u) * grad(v) * dx(definedon=self.mesh.Materials(region or self.region))
        a.Assemble()
        r, c, val = a.mat.COO()
        K = sp.csr_matrix((np.array(val), (np.array(r), np.array(c))), shape=(self.fes.ndof,) * 2)
        return (self.T.T @ K @ self.T).tocsr()

    # ------------------------------------------------------------------ solves
    def solve(self, rho_full: np.ndarray, sigma=None) -> State:
        K = self.stiffness_matrix(rho_full, sigma)
        Kuu = K[self.unknown][:, self.unknown].tocsc()
        rhs = self.f[self.unknown] - K[self.unknown][:, self.prescribed] @ self.x_prescribed[self.prescribed]
        self._lu = splu(Kuu)
        x = self.x_prescribed.copy()
        x[self.unknown] = self._lu.solve(rhs)
        return State(x=x, u=self.field(x),
                     port_voltage={p.name: x[self.n_interior + k] for k, p in enumerate(self.ports)})

    def adjoint(self, dJdx: np.ndarray) -> np.ndarray:
        """Solve K p = -dJ/dx on the unknowns (K is symmetric; reuses the last LU)."""
        p = np.zeros(self.n)
        p[self.unknown] = self._lu.solve(-dJdx[self.unknown])
        return p

    def field(self, x: np.ndarray) -> GridFunction:
        gf = GridFunction(self.fes)
        gf.vec.FV().NumPy()[:] = self.T @ x
        return gf

    def element_sensitivity(self, rho_full, x, p) -> np.ndarray:
        """dJ/drho_e = sigma'(rho_e) ∫_e ∇u·∇p dx, for every element."""
        gu, gp = self.field(x), self.field(p)
        g = np.array(Integrate(grad(gu) * grad(gp), self.mesh, element_wise=True))
        return self.interp.derivative(rho_full) * g * self.conducting

    def gradient(self, rho_full, state: State, dJdx: np.ndarray) -> np.ndarray:
        """Adjoint gradient of an objective J(x) with respect to the design densities."""
        p = self.adjoint(dJdx)
        return self.element_sensitivity(rho_full, state.x, p)[self.design_mask]

    # ------------------------------------------------------------------ post-processing
    def current_density(self, rho_full, state: State):
        self._sigma.vec.FV().NumPy()[:] = self.interp(rho_full)
        return -self._sigma * grad(state.u)

    def dissipated_power(self, rho_full, state: State) -> float:
        """Joule losses in the board plus power in the lumped resistors."""
        sig = GridFunction(L2(self.mesh, order=0))
        sig.vec.FV().NumPy()[:] = self.interp(rho_full)
        board = Integrate(sig * grad(state.u) * grad(state.u) * dx(definedon=self.mesh.Materials(self.region)), self.mesh)
        lumped = sum((state.port_voltage[r.a] - state.port_voltage[r.b]) ** 2 / r.resistance for r in self.resistors)
        return float(board + lumped)
