"""
Example: A kinesin-inspired 4-state motor under varying coupling regimes.

Demonstrates that one parameterization recovers:
  - Power-stroke behavior at large beta_couple (sharp commitment)
  - Brownian-ratchet behavior at small beta_couple
  - Hybrid behavior at intermediate beta_couple

Run:
    python examples/kinesin_4state.py
"""
import numpy as np
import matplotlib.pyplot as plt

from rhm import MotorParameters, RoyHybridModel


def make_kinesin_params(beta_couple: float = 1.0, F_ext: float = 0.0) -> MotorParameters:
    """Construct a simple 4-state kinesin-like parameter set (kT = 1 units)."""
    N = 4
    # 4 states evenly spaced along an 8 nm step (in arbitrary length units, here 8.0)
    L = 8.0
    x_star = np.array([0.0, 2.0, 4.0, 6.0])
    kappa = np.full(N, 5.0)
    G_chem = np.array([0.0, -2.0, -8.0, -10.0])  # downhill cycle
    # Forward rates dominate; backward rates small but nonzero (reversible)
    rate_matrix = np.zeros((N, N))
    forward = 50.0
    backward = 0.5
    for i in range(N):
        rate_matrix[i, (i + 1) % N] = forward
        rate_matrix[i, (i - 1) % N] = backward
    # Free energy injected by ATP hydrolysis (per cycle; here per transition i->i+1)
    delta_mu = np.zeros((N, N))
    for i in range(N):
        delta_mu[i, (i + 1) % N] = 5.0  # 20 kT total per cycle
        delta_mu[i, (i - 1) % N] = -5.0
    # Commitment positions slightly past the previous equilibrium
    x_commit = x_star + 0.5
    alpha = np.full(N, 1.0)
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
        gamma=1.0,
        kT=1.0,
        F_ext=F_ext,
    )


def main():
    rng = np.random.default_rng(seed=42)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    regimes = [
        ("Ratchet-like (β = 0.05)", 0.05),
        ("Hybrid (β = 1.0)", 1.0),
        ("Power-stroke-like (β = 10)", 10.0),
    ]
    for ax, (title, beta) in zip(axes, regimes):
        params = make_kinesin_params(beta_couple=beta)
        model = RoyHybridModel(params)
        result = model.simulate(t_max=10.0, x0=0.0, sigma0=0, dt=1e-3, rng=rng)
        ax.plot(result.t, result.x, lw=0.8)
        ax.set_title(f"{title}\nv = {result.mean_velocity:.2f}, cycles = {result.cycle_count}")
        ax.set_xlabel("Time")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Mechanical coordinate x")
    fig.suptitle("RHM trajectories across coupling regimes (kinesin-inspired 4-state model)")
    fig.tight_layout()
    fig.savefig("rhm_regimes.png", dpi=150)
    print("Wrote rhm_regimes.png")


if __name__ == "__main__":
    main()
