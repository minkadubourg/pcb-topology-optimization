import numpy as np
import pytest

from pcbtopo import SIMP, ConductionModel, DensityFilter, Port, Resistor
from pcbtopo.geometry import electrode_plate, resistor_board
from pcbtopo.interpolation import element_centroids
from pcbtopo.objectives import Compliance, FieldEnergy, Greyness, PortVoltageTarget, VolumeFraction
from pcbtopo.optimize import Problem


@pytest.fixture(scope="module")
def plate():
    return electrode_plate(maxh=0.1)


@pytest.fixture(scope="module")
def board():
    return resistor_board(maxh=0.08)


def resistor_model(mesh, interp=None, supply_voltage=True):
    p = Port("P", voltage=10.0) if supply_voltage else Port("P", current=1.0)
    return ConductionModel(mesh, [p, Port("N", voltage=0.0), Port("Rp"), Port("Rn")],
                           [Resistor("Rp", "Rn", 10.0)], interpolation=interp or SIMP(p=3))


def test_full_copper_plate_has_resistance_length_over_width(plate):
    # 2 x 2 plate, sigma = 1, electrodes on the full width: R = L / (sigma W) = 1
    m = ConductionModel(plate, [Port("top", current=1.0), Port("bottom", voltage=0.0)], interpolation=SIMP(p=1))
    s = m.solve(m.full_density(np.ones(m.n_design)))
    assert s.port_voltage["top"] == pytest.approx(1.0, rel=1e-10)


def test_series_bars_add_resistance(plate):
    # copper only in |x| < 0.5: width 1, R = 2; the empty part (1e-6) is a negligible shunt
    m = ConductionModel(plate, [Port("top", current=1.0), Port("bottom", voltage=0.0)], interpolation=SIMP(p=1))
    c = element_centroids(plate)[m.design_mask]
    s = m.solve(m.full_density((np.abs(c[:, 0]) < 0.5).astype(float)))
    assert s.port_voltage["top"] == pytest.approx(2.0, rel=0.05)


def test_power_balance(board):
    m = resistor_model(board, supply_voltage=False)
    rho = m.full_density(np.random.default_rng(1).uniform(0, 1, m.n_design))
    s = m.solve(rho)
    assert 1.0 * s.port_voltage["P"] == pytest.approx(m.dissipated_power(rho, s), rel=1e-9)


def test_isolated_resistor_is_driven_by_nothing(board):
    # empty board: P and N see 1e-6 conductance only; the resistor carries ~no voltage
    m = resistor_model(board, SIMP(sigma_min=1e-6, p=1))
    s = m.solve(m.full_density(np.zeros(m.n_design)))
    assert abs(s.port_voltage["Rp"] - s.port_voltage["Rn"]) < 1e-4


@pytest.mark.parametrize("p", [1, 3])
@pytest.mark.parametrize("objective", ["voltage", "compliance", "energy"])
def test_adjoint_gradient_matches_finite_differences(board, p, objective):
    """At the full 1e6 conductivity contrast: the gradient is exact, no need to
    use a different contrast in the physics and in the gradient."""
    m = resistor_model(board, SIMP(sigma_min=1e-6, p=p))
    obj = {"voltage": PortVoltageTarget(m, "Rp", "Rn", 5.0), "compliance": Compliance(m),
           "energy": FieldEnergy(m)}[objective]
    rng = np.random.default_rng(0)
    rho = rng.uniform(0.05, 0.95, m.n_design)
    full = m.full_density(rho)
    s = m.solve(full)
    J, dJdx = obj(s.x)
    g = m.gradient(full, s, dJdx)
    h = 1e-6
    for e in rng.choice(m.n_design, 4, replace=False):
        rp, rm = rho.copy(), rho.copy()
        rp[e] += h
        rm[e] -= h
        fd = (obj(m.solve(m.full_density(rp)).x)[0] - obj(m.solve(m.full_density(rm)).x)[0]) / (2 * h)
        assert g[e] == pytest.approx(fd, rel=1e-4, abs=1e-12 * max(abs(J), 1))


def test_filtered_problem_gradient_matches_finite_differences(board):
    m = resistor_model(board)
    areas = m.element_area[m.design_mask]
    prob = Problem(m, PortVoltageTarget(m, "Rp", "Rn", 10.0),
                   [VolumeFraction(areas, 0.3), Greyness(areas, 0.1)], filter_radius=0.15)
    rng = np.random.default_rng(2)
    rho = rng.uniform(0.1, 0.9, m.n_design)
    J, dJ, g, dg, _ = prob.evaluate(rho)
    h = 1e-6
    for e in rng.choice(m.n_design, 3, replace=False):
        rp, rm = rho.copy(), rho.copy()
        rp[e] += h
        rm[e] -= h
        Jp, _, gp, _, _ = prob.evaluate(rp)
        Jm, _, gm, _, _ = prob.evaluate(rm)
        assert dJ[e] == pytest.approx((Jp - Jm) / (2 * h), rel=1e-4)
        assert dg[:, e] == pytest.approx((gp - gm) / (2 * h), rel=1e-4)


def test_filter_preserves_constants_and_is_identity_at_zero_radius(board):
    m = resistor_model(board)
    c = element_centroids(board)[m.design_mask]
    a = m.element_area[m.design_mask]
    f = DensityFilter(c, a, 0.2)
    assert np.allclose(f(np.full(len(a), 0.7)), 0.7)
    assert np.allclose(DensityFilter(c, a, 0)(np.arange(len(a))), np.arange(len(a)))


def test_greyness_limit_above_one_quarter_can_never_be_active():
    a = np.ones(10)
    g, _ = Greyness(a, 0.4)(np.full(10, 0.5))     # worst case rho = 1/2
    assert g < 0


def test_ground_is_required(plate):
    with pytest.raises(ValueError):
        ConductionModel(plate, [Port("top", current=1.0), Port("bottom", current=-1.0)])


def test_unknown_pad_is_reported(plate):
    with pytest.raises(ValueError):
        ConductionModel(plate, [Port("top", voltage=0.0), Port("nowhere")])
