import numpy as np
import pytest

from pcbtopo import SIMP, ConductionModel, Port
from pcbtopo.geometry import electrode_plate, two_plate_board
from pcbtopo.harmonic import EPS0, MU0, solve_weakly_coupled
from pcbtopo.objectives import Compliance, VolumeFraction
from pcbtopo.optimize import Problem, mma


def test_mma_finds_the_resistance_bound_on_a_plate():
    """Minimum resistance between full-width electrodes with 30 % copper: any set of
    straight vertical bars of total width 0.6 is optimal, R* = 2 / 0.6."""
    mesh = electrode_plate(maxh=0.08)
    m = ConductionModel(mesh, [Port("top", current=1.0), Port("bottom", voltage=0.0)], interpolation=SIMP(p=3))
    areas = m.element_area[m.design_mask]
    prob = Problem(m, Compliance(m), [VolumeFraction(areas, 0.3)], filter_radius=0.1)
    # a perfectly uniform start is a stationary point here (every element has the same
    # sensitivity), so the start is perturbed slightly
    rho0 = 0.3 + 0.02 * np.random.default_rng(0).standard_normal(m.n_design)
    rho, hist = mma(prob, rho0, maxiter=80)
    rho_f = prob.filter(rho)
    R_bin = m.solve(m.full_density((rho_f > 0.5).astype(float))).port_voltage["top"]
    assert hist.objective[-1] < 0.5 * hist.objective[0]
    assert R_bin == pytest.approx(2 / 0.6, rel=0.05)
    assert hist.constraints[-1][0] < 1e-3


def test_mma_toy_problem():
    """Svanberg's toy problem, used to check the MMA wiring independently of the physics."""
    from mmapy import mmasub

    n, m = 3, 2
    x = np.array([[4.0], [3.0], [2.0]])
    xmin, xmax = np.zeros((n, 1)), 5 * np.ones((n, 1))
    xold1 = xold2 = x.copy()
    low, upp = xmin.copy(), xmax.copy()

    def f(x):
        x = x.ravel()
        f0 = x @ x
        g = np.array([[(x[0] - 5) ** 2 + (x[1] - 2) ** 2 + (x[2] - 1) ** 2 - 9],
                      [(x[0] - 3) ** 2 + (x[1] - 4) ** 2 + (x[2] - 3) ** 2 - 9]])
        dg = np.array([[2 * (x[0] - 5), 2 * (x[1] - 2), 2 * (x[2] - 1)],
                       [2 * (x[0] - 3), 2 * (x[1] - 4), 2 * (x[2] - 3)]])
        return f0, 2 * x.reshape(-1, 1), g, dg

    for it in range(1, 60):
        f0, df0, g, dg = f(x)
        xmma, *_, low, upp = mmasub(m, n, it, x, xmin, xmax, xold1, xold2, f0, df0, g, dg, low, upp,
                                     1.0, np.zeros((m, 1)), 1000 * np.ones((m, 1)), np.ones((m, 1)), 1.0)
        xold2, xold1, x = xold1, x, xmma
    assert x.ravel() == pytest.approx([2.0175, 1.7800, 1.2375], abs=2e-3)


@pytest.fixture(scope="module")
def harmonic():
    mesh = two_plate_board(maxh=0.12)
    sol = solve_weakly_coupled(mesh, {"core": 1.0, "plus": 1e6, "minus": 1e6}, {"top": 10.0, "bottom": -10j}, 10.0)
    return mesh, sol


def test_planes_are_equipotential_and_potential_is_linear_across_the_core(harmonic):
    mesh, sol = harmonic
    for z in (0.0, 0.025, 0.05, 0.075, 0.1):
        assert sol.V(mesh(0.4, 0.6, z)) == pytest.approx(-10j + (10 + 10j) * z / 0.1, abs=1e-5)
    assert sol.V(mesh(0.5, 0.5, 0.103)) == pytest.approx(10, abs=1e-4)
    assert sol.V(mesh(0.5, 0.5, -0.003)) == pytest.approx(-10j, abs=1e-4)


def test_vector_potential_is_linear_in_the_drive():
    mesh = two_plate_board(maxh=0.15)
    s1 = solve_weakly_coupled(mesh, {"core": 1.0, "plus": 1e6, "minus": 1e6}, {"top": 1.0, "bottom": 0.0}, 10.0)
    s2 = solve_weakly_coupled(mesh, {"core": 1.0, "plus": 1e6, "minus": 1e6}, {"top": 3.0, "bottom": 0.0}, 10.0)
    p = mesh(0.2, 0.4, 0.05)
    assert np.array(s2.B(p)) == pytest.approx(3 * np.array(s1.B(p)), rel=1e-8)


def test_ampere_law_in_the_core(harmonic):
    mesh, sol = harmonic
    half, z, n = 0.25, 0.05, 200
    s = np.linspace(-half, half, n, endpoint=False) + half / n
    dl = 2 * half / n
    circ = 0
    for si in s:
        circ += np.array(sol.B(mesh(0.5 + half, 0.5 + si, z)))[1] * dl
        circ -= np.array(sol.B(mesh(0.5 - half, 0.5 + si, z)))[1] * dl
        circ -= np.array(sol.B(mesh(0.5 + si, 0.5 + half, z)))[0] * dl
        circ += np.array(sol.B(mesh(0.5 + si, 0.5 - half, z)))[0] * dl
    Ez = -(10 + 10j) / 0.1
    I = (1.0 + 1j * 10 * EPS0) * Ez * (2 * half) ** 2
    assert abs(circ - MU0 * I) / abs(MU0 * I) < 0.02
