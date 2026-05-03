# Roy Hybrid Model (RHM)

A unified mechanochemical framework for protein molecular motors, reconciling the
**power-stroke** and **Brownian-ratchet** schools through a tunable coupling function
on a 2D free-energy landscape.

This repository contains the reference implementation of the Roy Hybrid Model and
the **RHM-SSA** simulator (a hybrid Gillespie–Langevin scheme) accompanying:

> Roy, K. R. *The Roy Hybrid Model: A Unified Mechanochemical Framework Reconciling
> Power Stroke and Brownian Ratchet Mechanisms in Protein Molecular Motors.*

## What it does

RHM treats the mechanical coordinate `x` and chemical state `σ` of a motor protein
on a 2D energy landscape. A coupling function

```
λ(σ, x) ∈ [0, 1]
```

interpolates between two limits of one model:

- `λ → 1`: **power-stroke** behavior (chemistry deterministically drives mechanics)
- `λ → 0`: **Brownian-ratchet** behavior (chemistry biases a thermal-fluctuation landscape)

Intermediate `λ` produces **hybrid strokes** — composite events with a diffusive search
phase, a chemistry-driven commitment, and a deterministic relaxation. The framework
applies to ribosomal translocation, myosin, kinesin, and F₁-ATPase under one parameter set.

## Installation

```bash
git clone https://github.com/kroy3/roy-hybrid-model.git
cd roy-hybrid-model
pip install -e .
```

Requires Python ≥ 3.9, NumPy, and (for examples) Matplotlib.

## Quick start

```python
import numpy as np
from rhm import MotorParameters, RoyHybridModel

# A minimal 2-state motor
params = MotorParameters(
    n_states=2,
    x_star=np.array([0.0, 1.0]),
    kappa=np.full(2, 5.0),
    G_chem=np.zeros(2),
    rate_matrix=np.array([[0.0, 10.0], [10.0, 0.0]]),
    delta_mu=np.array([[0.0, 2.0], [-2.0, 0.0]]),
    x_commit=np.array([0.5, 0.5]),
    alpha=np.array([1.0, 1.0]),
    beta_couple=1.0,  # Hybrid regime
)

model = RoyHybridModel(params)
result = model.simulate(t_max=10.0, dt=1e-3)

print(f"Mean velocity: {result.mean_velocity:.3f}")
print(f"Chemical cycles completed: {result.cycle_count}")
```

## The three regimes

Run `python examples/kinesin_4state.py` to reproduce trajectories across the three
coupling regimes for a kinesin-inspired 4-state motor:

- Ratchet-like (β = 0.05): diffusive
- Hybrid (β = 1): mixed dynamics
- Power-stroke-like (β = 10): plateau-then-step trajectories

Output is saved to `rhm_regimes.png`.

## Algorithm overview

The simulator implements **RHM-SSA** (Algorithm 1 of the paper):

1. Compute the effective potential `V_eff(x; σ) = λ U_σ + (1−λ) Ū`.
2. Compute the position-dependent total chemical exit rate `K(x)`.
3. Draw the next chemical event time τ ~ Exp(K) (Gillespie).
4. Evolve `x` by Euler–Maruyama on `V_eff` until min(τ, Δt).
5. If a chemical event fires, choose the target state by rate-weighted probabilities.
6. Repeat.

Local detailed balance is enforced through the symmetric Kramers-like rate splitting
(see `rhm/model.py`).

## Repository structure

```
roy-hybrid-model/
├── rhm/
│   ├── __init__.py
│   └── model.py          # Core: MotorParameters, RoyHybridModel, RHM-SSA
├── examples/
│   └── kinesin_4state.py # Three-regime demonstration
├── tests/
│   └── test_model.py     # Validation tests (pytest)
├── pyproject.toml
├── LICENSE
└── README.md
```

## Validation

Run the test suite:

```bash
pytest tests/
```

Tests verify:
- Coupling function bounded in [0, 1]
- Correct limits at small/large β
- Zero net velocity at Δμ = 0 (equilibrium)
- Force continuity
- End-to-end simulation runs

## Citation

If you use this code, please cite:

```bibtex
@article{Roy2026RHM,
  title={The Roy Hybrid Model: A Unified Mechanochemical Framework
         Reconciling Power Stroke and Brownian Ratchet Mechanisms
         in Protein Molecular Motors},
  author={Roy, Kushal Raj},
  journal={Submitted},
  year={2026}
}
```

## License

MIT License — see [LICENSE](LICENSE).

## Contact

Kushal Raj Roy — kushalrajroy1@gmail.com
