r"""Time-harmonic A–V model of a board (3D), weakly coupled.

With fields :math:`\propto e^{j\omega t}`, Ohm's law :math:`J = \sigma E`,
:math:`E = -j\omega A - \nabla V` and :math:`B = \nabla\times A`, Maxwell's equations give

.. math::
    \nabla\times(\nu\nabla\times A) + (j\omega\sigma - \omega^2\varepsilon) A
        + (\sigma + j\omega\varepsilon)\nabla V = J_\text{free},
    \qquad
    \nabla\cdot\big(\varepsilon(\nabla V + j\omega A)\big) = -\rho .

In the quasi-static regime the feedback of the induced field on the scalar
potential is neglected (the :math:`j\omega A` terms in the equation for V),
which decouples the system into two solves done one after the other:

1. conservation of the total (conduction + displacement) current,
   :math:`\int (\sigma + j\omega\varepsilon)\,\nabla V\cdot\nabla V^*\,d\Omega = 0`, with the
   electrode voltages imposed (complex amplitudes, so electrodes can be out of
   phase). Using :math:`\sigma + j\omega\varepsilon` rather than :math:`\varepsilon` alone
   makes conductors equipotential, as they should be;
2. :math:`\int \nu\,\nabla\times A\cdot\nabla\times A^* + (j\omega\sigma-\omega^2\varepsilon)A\cdot A^*
   = \int J_\text{free}\cdot A^* - \int(\sigma+j\omega\varepsilon)\nabla V\cdot A^*`,
   with :math:`n\times A = 0` on the outer boundary.

The fully coupled system is non-symmetric and needs a gauge; the weakly
coupled one is two symmetric (complex) problems solved with a direct solver
in seconds rather than hours.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ngsolve import (H1, BilinearForm, CoefficientFunction, GridFunction, HCurl, InnerProduct,
                     LinearForm, curl, dx, grad)

MU0 = 4e-7 * np.pi
EPS0 = 8.854e-12


@dataclass
class HarmonicSolution:
    V: GridFunction
    A: GridFunction
    omega: float

    @property
    def E(self):
        return -grad(self.V) - 1j * self.omega * self.A

    @property
    def B(self):
        return curl(self.A)


def solve_weakly_coupled(mesh, sigma: dict, electrodes: dict, omega: float,
                         eps_r: dict | None = None, J_free=None, order: int = 1,
                         outer: str = "outer") -> HarmonicSolution:
    """sigma, eps_r: {material: value}; electrodes: {boundary: complex voltage}."""
    sig = CoefficientFunction([sigma.get(m, 0.0) for m in mesh.GetMaterials()])
    eps = EPS0 * CoefficientFunction([(eps_r or {}).get(m, 1.0) for m in mesh.GetMaterials()])
    nu = 1 / MU0

    # 1) scalar potential
    fv = H1(mesh, order=order, complex=True, dirichlet="|".join(electrodes))
    v, vs = fv.TnT()
    av = BilinearForm((sig + 1j * omega * eps) * grad(v) * grad(vs) * dx).Assemble()
    V = GridFunction(fv)
    V.Set(mesh.BoundaryCF(electrodes, default=0), definedon=mesh.Boundaries("|".join(electrodes)))
    r = -(av.mat * V.vec)
    V.vec.data += av.mat.Inverse(fv.FreeDofs(), inverse="sparsecholesky") * r

    # 2) vector potential driven by the conduction + displacement current of V
    fa = HCurl(mesh, order=order, complex=True, dirichlet=outer)
    a, as_ = fa.TnT()
    aa = BilinearForm(fa, symmetric=True)
    aa += nu * curl(a) * curl(as_) * dx
    aa += (1j * omega * sig - omega ** 2 * eps) * a * as_ * dx
    aa.Assemble()
    f = LinearForm(fa)
    f += -(sig + 1j * omega * eps) * InnerProduct(grad(V), as_) * dx
    if J_free is not None:
        f += InnerProduct(J_free, as_) * dx
    f.Assemble()
    A = GridFunction(fa)
    A.vec.data = aa.mat.Inverse(fa.FreeDofs(), inverse="sparsecholesky") * f.vec
    return HarmonicSolution(V=V, A=A, omega=omega)
