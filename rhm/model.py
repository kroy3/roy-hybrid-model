"""
Roy Hybrid Model (RHM) for protein molecular motors.

A unified mechanochemical framework reconciling the power-stroke and
Brownian-ratchet pictures via a tunable coupling function on a 2D landscape.

Reference: Roy, K. R. "The Roy Hybrid Model: A Unified Mechanochemical
Framework Reconciling Power Stroke and Brownian Ratchet Mechanisms in
Protein Molecular Motors."
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

__all__ = ["MotorParameters", "RoyHybridModel", "SimulationResult"]


@dataclass
class MotorParameters:
    """
    Parameters of a mechanochemical motor in the Roy Hybrid framework.

    Attributes
    ----------
    n_states : int
        Number of discrete chemical states N.
    x_star : np.ndarray, shape (n_states,)
        Equilibrium mechanical position x*_sigma in each chemical state.
    kappa : np.ndarray, shape (n_states,)
        Harmonic stiffness in each chemical state. The bare conformational
        potential is U^(0)_sigma(x) = 0.5 * kappa[sigma] * (x - x_star[sigma])**2.
    G_chem : np.ndarray, shape (n_states,)
        Chemical free-energy offset of each state (in units of kT).
    rate_matrix : np.ndarray, shape (n_states, n_states)
        Bare (zero-load) chemical transition rate matrix k0[i, j] = k_{i->j}.
        Diagonal entries are ignored.
    delta_mu : np.ndarray, shape (n_states, n_states)
        Chemical free-energy injected per transition i -> j (in units of kT).
        Used by local-detailed-balance bookkeeping.
    x_commit : np.ndarray, shape (n_states,)
        Commitment position x^c_sigma for the sigmoidal coupling function.
    alpha : np.ndarray, shape (n_states,)
        Coupling-sharpness parameter alpha_sigma (units: 1/length).
    beta_couple : float
        Dimensionless coupling exponent in lambda(sigma, x). beta -> 0 gives
        loose (ratchet) coupling everywhere; large beta gives sharp commitment.
    gamma : float
        Mechanical friction coefficient.
    kT : float
        Thermal energy (default 1.0; sets the energy scale).
    F_ext : float
        External load (force opposing motion). Default 0.
    """

    n_states: int
    x_star: np.ndarray
    kappa: np.ndarray
    G_chem: np.ndarray
    rate_matrix: np.ndarray
    delta_mu: np.ndarray
    x_commit: np.ndarray
    alpha: np.ndarray
    beta_couple: float = 1.0
    gamma: float = 1.0
    kT: float = 1.0
    F_ext: float = 0.0

    def __post_init__(self) -> None:
        N = self.n_states
        for name, arr in [
            ("x_star", self.x_star),
            ("kappa", self.kappa),
            ("G_chem", self.G_chem),
            ("x_commit", self.x_commit),
            ("alpha", self.alpha),
        ]:
            if arr.shape != (N,):
                raise ValueError(f"{name} must have shape ({N},), got {arr.shape}")
        if self.rate_matrix.shape != (N, N):
            raise ValueError(f"rate_matrix must have shape ({N},{N})")
        if self.delta_mu.shape != (N, N):
            raise ValueError(f"delta_mu must have shape ({N},{N})")


@dataclass
class SimulationResult:
    """Output of an RHM-SSA trajectory simulation."""

    t: np.ndarray
    x: np.ndarray
    sigma: np.ndarray
    chemical_event_times: np.ndarray
    chemical_event_transitions: np.ndarray  # array of (sigma_old, sigma_new)
    parameters: MotorParameters

    @property
    def mean_velocity(self) -> float:
        """Average mechanical velocity across the trajectory."""
        if len(self.t) < 2:
            return 0.0
        return float((self.x[-1] - self.x[0]) / (self.t[-1] - self.t[0]))

    @property
    def cycle_count(self) -> int:
        """Number of completed chemical cycles (transitions of any kind)."""
        return int(len(self.chemical_event_times))


class RoyHybridModel:
    """
    The Roy Hybrid Model and its associated simulator (RHM-SSA).

    The core dynamical equation is:

        gamma * dx = -dV_eff/dx * dt + sqrt(2 * gamma * kT) * dW

    where the effective potential is

        V_eff(x; sigma) = lambda(sigma, x) * U_sigma(x)
                          + (1 - lambda(sigma, x)) * U_avg(x)

    The coupling function lambda interpolates between tight (power stroke,
    lambda -> 1) and loose (Brownian ratchet, lambda -> 0) coupling.

    Chemical transitions occur with position-dependent Bell-type rates:

        k_{sigma -> sigma'}(x) = k0_{sigma -> sigma'} *
                                 exp(-Delta_U_dagger(x) / kT)
    """

    def __init__(self, params: MotorParameters):
        self.p = params

    # ---- Potentials ----

    def U_bare(self, x: float | np.ndarray, sigma: int) -> float | np.ndarray:
        """Bare conformational potential U_sigma(x), excluding external load."""
        p = self.p
        return 0.5 * p.kappa[sigma] * (x - p.x_star[sigma]) ** 2 + p.G_chem[sigma] * p.kT

    def U_state(self, x: float | np.ndarray, sigma: int) -> float | np.ndarray:
        """Full state potential including external load: U_sigma(x) - F_ext * x."""
        return self.U_bare(x, sigma) - self.p.F_ext * x

    def U_avg(self, x: float | np.ndarray) -> float | np.ndarray:
        """Chemistry-averaged baseline potential (the 'ratchet baseline')."""
        return np.mean([self.U_state(x, s) for s in range(self.p.n_states)], axis=0)

    def coupling(self, x: float | np.ndarray, sigma: int) -> float | np.ndarray:
        """
        Sigmoidal coupling lambda(sigma, x) in [0, 1].

        lambda(sigma, x) = 1 / (1 + exp(-beta * kT * alpha_sigma * (x - x^c_sigma)))
        """
        p = self.p
        z = p.beta_couple * p.kT * p.alpha[sigma] * (x - p.x_commit[sigma])
        # numerically stable sigmoid
        return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))

    def V_eff(self, x: float | np.ndarray, sigma: int) -> float | np.ndarray:
        """The Roy effective potential."""
        lam = self.coupling(x, sigma)
        return lam * self.U_state(x, sigma) + (1.0 - lam) * self.U_avg(x)

    def force(self, x: float, sigma: int) -> float:
        """
        Mechanical force f = -dV_eff/dx at (x, sigma).

        Computed analytically via the product rule on V_eff = lambda*U + (1-lambda)*Ubar.
        """
        p = self.p
        # Components
        U_s = self.U_state(x, sigma)
        Ubar = self.U_avg(x)
        lam = self.coupling(x, sigma)

        # dU_s/dx = kappa_sigma * (x - x_star_sigma) - F_ext
        dU_s = p.kappa[sigma] * (x - p.x_star[sigma]) - p.F_ext
        # dUbar/dx is the average of dU_s'/dx over states
        dUbar = np.mean(
            [p.kappa[s] * (x - p.x_star[s]) - p.F_ext for s in range(p.n_states)]
        )
        # dlambda/dx
        z = p.beta_couple * p.kT * p.alpha[sigma] * (x - p.x_commit[sigma])
        dlam = (
            p.beta_couple * p.kT * p.alpha[sigma] * lam * (1.0 - lam)
        )  # standard sigmoid derivative

        dV = dlam * (U_s - Ubar) + lam * dU_s + (1.0 - lam) * dUbar
        return float(-dV)

    # ---- Chemistry ----

    def transition_rate(self, x: float, sigma_from: int, sigma_to: int) -> float:
        """
        Position-dependent Bell-type rate k_{sigma -> sigma'}(x).

        For simplicity here we adopt a load-modulated form
            k(x) = k0 * exp(-(V_eff(x, sigma') - V_eff(x, sigma) - delta_mu) / (2 kT))
        which automatically respects local detailed balance with its reverse.
        """
        if sigma_from == sigma_to:
            return 0.0
        p = self.p
        k0 = p.rate_matrix[sigma_from, sigma_to]
        if k0 <= 0:
            return 0.0
        dV = self.V_eff(x, sigma_to) - self.V_eff(x, sigma_from)
        dmu = p.delta_mu[sigma_from, sigma_to] * p.kT
        # symmetric Kramers-like splitting; ensures detailed balance with reverse
        return k0 * np.exp(-(dV - dmu) / (2.0 * p.kT))

    def total_exit_rate(self, x: float, sigma: int) -> float:
        return sum(
            self.transition_rate(x, sigma, s_to) for s_to in range(self.p.n_states)
        )

    # ---- RHM-SSA simulation ----

    def simulate(
        self,
        t_max: float,
        x0: float = 0.0,
        sigma0: int = 0,
        dt: float = 1e-3,
        record_every: int = 1,
        rng: Optional[np.random.Generator] = None,
    ) -> SimulationResult:
        """
        Hybrid Gillespie--Langevin simulation (RHM-SSA, Algorithm 1 of the paper).

        Parameters
        ----------
        t_max : float
            Total simulation time.
        x0, sigma0 : initial mechanical and chemical states.
        dt : float
            Mechanical timestep. Should satisfy dt < gamma / kappa_max for stability.
        record_every : int
            Down-sampling factor for the recorded trajectory.
        rng : np.random.Generator, optional
            Random number generator. Defaults to default_rng().
        """
        if rng is None:
            rng = np.random.default_rng()

        p = self.p
        sqrt_2DkT_dt = np.sqrt(2.0 * p.kT * dt / p.gamma)

        # Storage
        n_steps = int(np.ceil(t_max / dt))
        n_record = n_steps // record_every + 1
        t_rec = np.empty(n_record)
        x_rec = np.empty(n_record)
        sigma_rec = np.empty(n_record, dtype=np.int64)
        idx = 0
        t_rec[0] = 0.0
        x_rec[0] = x0
        sigma_rec[0] = sigma0
        idx += 1

        chem_times: list[float] = []
        chem_trans: list[tuple[int, int]] = []

        # State
        t = 0.0
        x = x0
        sigma = sigma0

        # Pre-draw next chemical event
        K = self.total_exit_rate(x, sigma)
        tau_chem = -np.log(rng.uniform()) / K if K > 0 else np.inf
        t_next_chem = t + tau_chem

        for step in range(1, n_steps + 1):
            # Langevin step
            f = self.force(x, sigma)
            x = x + (f / p.gamma) * dt + sqrt_2DkT_dt * rng.standard_normal()
            t = step * dt

            # Has a chemical event fired in this dt?
            if t >= t_next_chem:
                # Choose target state by current rate-weighted probabilities
                rates = np.array(
                    [self.transition_rate(x, sigma, s) for s in range(p.n_states)]
                )
                total = rates.sum()
                if total > 0:
                    probs = rates / total
                    sigma_new = int(rng.choice(p.n_states, p=probs))
                    chem_times.append(t)
                    chem_trans.append((sigma, sigma_new))
                    sigma = sigma_new

                # Re-draw next chemical event
                K = self.total_exit_rate(x, sigma)
                tau_chem = -np.log(rng.uniform()) / K if K > 0 else np.inf
                t_next_chem = t + tau_chem

            # Record
            if step % record_every == 0 and idx < n_record:
                t_rec[idx] = t
                x_rec[idx] = x
                sigma_rec[idx] = sigma
                idx += 1

        return SimulationResult(
            t=t_rec[:idx],
            x=x_rec[:idx],
            sigma=sigma_rec[:idx],
            chemical_event_times=np.array(chem_times),
            chemical_event_transitions=np.array(chem_trans, dtype=np.int64).reshape(
                -1, 2
            )
            if chem_trans
            else np.zeros((0, 2), dtype=np.int64),
            parameters=p,
        )
