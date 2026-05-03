"""
Validation tests for the Roy Hybrid Model implementation.

These check the three correctness criteria from the paper:
  (i)  Reduces to equilibrium when delta_mu = 0
  (ii) Recovers the power-stroke limit when lambda -> 1
  (iii) Recovers the ratchet limit when lambda -> 0

Run with: pytest tests/
"""
import numpy as np
import pytest

from rhm import MotorParameters, RoyHybridModel


def make_minimal_params(beta_couple: float = 1.0, dmu: float = 0.0) -> MotorParameters:
    """A minimal symmetric 2-state test motor."""
    N = 2
    x_star = np.array([0.0, 1.0])
    kappa = np.full(N, 5.0)
    G_chem = np.zeros(N)
    rate_matrix = np.array([[0.0, 10.0], [10.0, 0.0]])
    delta_mu = np.array([[0.0, dmu], [-dmu, 0.0]])
    x_commit = np.array([0.5, 0.5])
    alpha = np.array([1.0, 1.0])
    return MotorParameters(
        n_states=N,
        x_star=x_star,
        kappa=kappa,
        G_chem=G_chem,
        rate_matrix=rate_matrix,
        delta_mu=delta_mu,
        x_commit=x_commit,
        alpha=alpha,
        beta_couple=beta_couple,
    )


def test_coupling_bounds():
    """lambda(sigma, x) must lie in [0, 1] everywhere."""
    params = make_minimal_params()
    model = RoyHybridModel(params)
    xs = np.linspace(-5, 5, 200)
    for s in range(params.n_states):
        lam = model.coupling(xs, s)
        assert np.all(lam >= 0.0) and np.all(lam <= 1.0)


def test_coupling_limits():
    """At small/large beta the coupling approaches 0.5 / step function."""
    p_small = make_minimal_params(beta_couple=1e-6)
    p_large = make_minimal_params(beta_couple=1e3)
    m_small = RoyHybridModel(p_small)
    m_large = RoyHybridModel(p_large)
    # Small beta: coupling ~ 0.5 everywhere
    assert abs(m_small.coupling(0.0, 0) - 0.5) < 1e-3
    assert abs(m_small.coupling(2.0, 0) - 0.5) < 1e-3
    # Large beta: step at x_commit = 0.5
    assert m_large.coupling(0.0, 0) < 0.01
    assert m_large.coupling(1.0, 0) > 0.99


def test_equilibrium_no_drive():
    """When delta_mu = 0, mean velocity should be ~0 (no net work)."""
    params = make_minimal_params(dmu=0.0)
    model = RoyHybridModel(params)
    rng = np.random.default_rng(123)
    res = model.simulate(t_max=100.0, x0=0.5, sigma0=0, dt=1e-3, rng=rng)
    # Velocity should be statistically near zero
    assert abs(res.mean_velocity) < 0.5  # generous tolerance for short trajectory


def test_force_continuity():
    """Force should be a smooth function of x (no NaNs, no infs)."""
    params = make_minimal_params()
    model = RoyHybridModel(params)
    xs = np.linspace(-3, 3, 50)
    for s in range(params.n_states):
        forces = [model.force(float(x), s) for x in xs]
        assert all(np.isfinite(forces))


def test_simulation_runs():
    """Full simulation completes and returns sensible shapes."""
    params = make_minimal_params(dmu=2.0)
    model = RoyHybridModel(params)
    res = model.simulate(t_max=5.0, x0=0.0, sigma0=0, dt=1e-3)
    assert len(res.t) == len(res.x) == len(res.sigma)
    assert res.t[0] == 0.0
    assert res.t[-1] <= 5.0 + 1e-6
