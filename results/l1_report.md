# L1 Computational Model: Environment-Shaped Exploration and Richness

## What changed relative to the earlier bandit simulations

This is a computational-level model. The agent does not learn an action policy online. Instead, the code computes the optimal finite-horizon policy exactly by backward induction over a discretized exposure state.

The output is not a single trajectory. The main output is a comparative-statics surface: richness as a function of environment structure (`sigma`) and enrichment/security (`E`).

## Model

- Horizon: T = 12
- Monte Carlo rollouts per grid point: 1500
- Arms: K = 5 experiences.
- Exposure state: h_k in {0, 1, 2, 3, 4}.
- Confidence/routinization curve: c_k(h) = 1 - exp(-gamma_k h).
- Novelty: 1 - c_k(h) = exp(-gamma_k h).
- `sigma`: probability that exposure persists after choosing an arm. High sigma means structure/stationarity and durable routinization. Low sigma means drift/refresh.
- `E`: enrichment/security. Low E penalizes risky/ambiguous arms.

## Objective functions

Two objectives were tested because the objective is the key modeling decision.

1. `novelty`: reward equals current novelty, exp(-gamma_k h). Hard-to-learn arms remain novel, so the optimum tends to embrace them.
2. `learning_progress`: reward equals delta confidence, c_k(h+1) - c_k(h). Hard-to-learn arms give less progress per step, so the optimum can drop them, especially under risk penalty.

## Main diagnostics: learning_progress objective

| Comparison | Flow richness | Stock richness | Combined flow/stock |
|---|---:|---:|---:|
| Drift, high enrichment: sigma=0, E=1 | 1.000 | 1.00 | 0.600 |
| Stationary, high enrichment: sigma=1, E=1 | 0.698 | 5.00 | 0.849 |
| Mid-structure, deprivation: sigma=0.5, E=0 | 0.662 | 1.51 | 0.482 |
| Mid-structure, enrichment: sigma=0.5, E=1 | 0.844 | 3.64 | 0.786 |

The best combined flow/stock point under `learning_progress` is sigma=0.83, E=1.00, combined=0.866.

## Extreme-environment policy diagnostics

| Objective | Environment | Flow | Stock | Diagnostic action path |
|---|---|---:|---:|---|
| learning_progress | stationary | 0.698 | 5.00 | 0 1 0 0 2 2 3 3 4 1 1 4 |
| learning_progress | drifting | 1.000 | 1.00 | 4 4 4 4 4 4 4 4 4 4 4 4 |
| novelty | stationary | 0.707 | 5.00 | 0 1 1 3 3 0 0 0 1 2 2 4 |
| novelty | drifting | 1.000 | 1.00 | 0 0 0 0 0 0 0 0 0 0 0 0 |

## Interpretation

- P1, flow under drift: lower sigma refreshes exposure, so current novelty remains higher.
- P2, learnability conversion loss: under `learning_progress`, hard-to-learn/risky arms are less attractive because they yield less confidence gain per step and can be priced out by low enrichment.
- P3, deprivation retreat: lowering E reduces both flow and stock by penalizing risky/ambiguous arms.
- P4, sweet spot: the combined readout is strongest where enrichment is high and structure is not simply maximal. The exact peak depends on the chosen objective.

The objective-function comparison is central. If the objective is raw novelty, hard-to-learn arms stay attractive because they stay fresh. If the objective is learning progress, the same arms can be avoided because they do not convert exposure into confidence efficiently. This is the main modeling hinge.

## Outputs

- `results/l1_surface_results.csv`
- `results/l1_policy_diagnostics.csv`
- `results/l1_learning_progress_surfaces.png`
- `results/l1_novelty_surfaces.png`
- `results/l1_objective_comparison.png`
