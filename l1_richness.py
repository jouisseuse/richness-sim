#!/usr/bin/env python3
"""L1 computational-level model for environment-shaped richness.

This file differs from the earlier bandit simulations:
- the agent does not learn an action policy online;
- we compute the optimal finite-horizon policy exactly by backward induction;
- predictions are comparative-static surfaces over environment parameters.

Core knobs:
- sigma: structure / stationarity. High sigma makes exposure persist and routinize.
- E: enrichment / security. Low E penalizes risky/ambiguous arms.
- objective: either current novelty or learning progress.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class RichnessEnv:
    K: int = 5
    H: int = 4
    gammas: tuple[float, ...] | None = None
    sigma: float = 1.0
    E: float = 1.0
    risk: tuple[float, ...] | None = None
    lam_max: float = 1.5
    objective: str = "learning_progress"

    def __post_init__(self) -> None:
        gammas = np.asarray(self.gammas if self.gammas is not None else np.linspace(0.3, 1.2, self.K))
        risk = np.asarray(self.risk if self.risk is not None else (1.0 - gammas / gammas.max()))
        object.__setattr__(self, "gamma_arr", gammas)
        object.__setattr__(self, "risk_arr", risk)

    def confidence(self, k: int, h: int) -> float:
        return float(1.0 - np.exp(-self.gamma_arr[k] * h))

    def novelty(self, k: int, h: int) -> float:
        return float(np.exp(-self.gamma_arr[k] * h))

    def base_reward(self, k: int, h: int) -> float:
        if self.objective == "novelty":
            return self.novelty(k, h)
        if self.objective == "learning_progress":
            return self.confidence(k, h + 1) - self.confidence(k, h)
        raise ValueError(f"Unknown objective: {self.objective}")

    def reward(self, k: int, h: int) -> float:
        penalty = self.lam_max * (1.0 - self.E) * self.risk_arr[k]
        return float(self.base_reward(k, h) - penalty)

    def transitions(self, h: int) -> tuple[tuple[int, float], tuple[int, float]]:
        return ((min(h + 1, self.H), self.sigma), (0, 1.0 - self.sigma))


def enumerate_states(K: int, H: int) -> tuple[list[tuple[int, ...]], dict[tuple[int, ...], int]]:
    states = list(product(range(H + 1), repeat=K))
    return states, {state: i for i, state in enumerate(states)}


def solve_optimal_policy(env: RichnessEnv, T: int = 15) -> tuple[np.ndarray, dict[tuple[int, ...], int]]:
    states, idx = enumerate_states(env.K, env.H)
    value = np.zeros(len(states))
    policy = np.zeros((T, len(states)), dtype=np.int16)

    for t in reversed(range(T)):
        new_value = np.empty(len(states))
        for state_index, state in enumerate(states):
            best_value = -1e18
            best_arm = 0
            for k in range(env.K):
                q = env.reward(k, state[k])
                for next_h, prob in env.transitions(state[k]):
                    next_state = state[:k] + (next_h,) + state[k + 1 :]
                    q += prob * value[idx[next_state]]
                if q > best_value:
                    best_value = q
                    best_arm = k
            new_value[state_index] = best_value
            policy[t, state_index] = best_arm
        value = new_value

    return policy, idx


def measure_richness(
    env: RichnessEnv,
    policy: np.ndarray,
    idx: dict[tuple[int, ...], int],
    T: int = 15,
    n_rollouts: int = 1500,
    seed: int = 0,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    flow = np.zeros(n_rollouts)
    stock = np.zeros(n_rollouts)
    value = np.zeros(n_rollouts)

    for i in range(n_rollouts):
        state = (0,) * env.K
        visited_high_novelty: set[int] = set()
        novelty_sum = 0.0
        reward_sum = 0.0
        for t in range(T):
            k = int(policy[t, idx[state]])
            nov = env.novelty(k, state[k])
            novelty_sum += nov
            reward_sum += env.reward(k, state[k])
            if nov > 0.5:
                visited_high_novelty.add(k)

            next_h = min(state[k] + 1, env.H) if rng.random() < env.sigma else 0
            state = state[:k] + (next_h,) + state[k + 1 :]

        flow[i] = novelty_sum / T
        stock[i] = len(visited_high_novelty)
        value[i] = reward_sum / T

    return float(flow.mean()), float(stock.mean()), float(value.mean())


def initial_state_action_sequence(env: RichnessEnv, policy: np.ndarray, idx: dict[tuple[int, ...], int], T: int) -> list[int]:
    """Deterministic diagnostic path using expected-persistence threshold."""
    state = (0,) * env.K
    actions: list[int] = []
    for t in range(T):
        k = int(policy[t, idx[state]])
        actions.append(k)
        next_h = min(state[k] + 1, env.H) if env.sigma >= 0.5 else 0
        state = state[:k] + (next_h,) + state[k + 1 :]
    return actions


def sweep(
    objective: str = "learning_progress",
    T: int = 12,
    n_rollouts: int = 1500,
    sigmas: np.ndarray | None = None,
    Es: np.ndarray | None = None,
    seed: int = 20260707,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, float | str]]]:
    sigmas = np.linspace(0.0, 1.0, 7) if sigmas is None else sigmas
    Es = np.linspace(0.0, 1.0, 7) if Es is None else Es
    flow = np.zeros((len(sigmas), len(Es)))
    stock = np.zeros_like(flow)
    avg_value = np.zeros_like(flow)
    rows: list[dict[str, float | str]] = []

    for i, sigma in enumerate(sigmas):
        for j, E in enumerate(Es):
            env = RichnessEnv(sigma=float(sigma), E=float(E), objective=objective)
            policy, idx = solve_optimal_policy(env, T=T)
            f, s, v = measure_richness(env, policy, idx, T=T, n_rollouts=n_rollouts, seed=seed + i * 100 + j)
            flow[i, j] = f
            stock[i, j] = s
            avg_value[i, j] = v
            rows.append(
                {
                    "objective": objective,
                    "sigma": float(sigma),
                    "E": float(E),
                    "flow": f,
                    "stock": s,
                    "stock_share": s / env.K,
                    "avg_objective_value": v,
                    "combined_flow_stock": 0.5 * f + 0.5 * (s / env.K),
                }
            )

    return sigmas, Es, flow, stock, avg_value, rows


def write_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_surface_pair(
    objective: str,
    sigmas: np.ndarray,
    Es: np.ndarray,
    flow: np.ndarray,
    stock: np.ndarray,
    outdir: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True)
    for axis, matrix, title in (
        (axes[0], flow, "Flow richness"),
        (axes[1], stock, "Stock richness"),
    ):
        image = axis.imshow(matrix, origin="lower", aspect="auto", cmap="viridis")
        axis.set_title(f"{title}: {objective}")
        axis.set_xlabel("Enrichment E")
        axis.set_ylabel("Structure sigma")
        axis.set_xticks(range(len(Es)))
        axis.set_xticklabels([f"{x:.2g}" for x in Es], rotation=45)
        axis.set_yticks(range(len(sigmas)))
        axis.set_yticklabels([f"{x:.2g}" for x in sigmas])
        fig.colorbar(image, ax=axis, shrink=0.85)
    fig.savefig(outdir / f"l1_{objective}_surfaces.png", dpi=180)
    plt.close(fig)


def plot_objective_comparison(rows: list[dict[str, float | str]], outdir: Path) -> None:
    objectives = ["learning_progress", "novelty"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True, sharey=True)
    for axis, objective in zip(axes, objectives):
        subset = [row for row in rows if row["objective"] == objective and float(row["E"]) == 1.0]
        subset = sorted(subset, key=lambda row: float(row["sigma"]))
        axis.plot(
            [float(row["sigma"]) for row in subset],
            [float(row["flow"]) for row in subset],
            marker="o",
            label="flow",
            color="#4C78A8",
        )
        axis.plot(
            [float(row["sigma"]) for row in subset],
            [float(row["stock_share"]) for row in subset],
            marker="o",
            label="stock/K",
            color="#F58518",
        )
        axis.plot(
            [float(row["sigma"]) for row in subset],
            [float(row["combined_flow_stock"]) for row in subset],
            marker="o",
            label="combined",
            color="#54A24B",
        )
        axis.set_title(f"Objective: {objective}")
        axis.set_xlabel("sigma")
        axis.grid(True, alpha=0.25)
        axis.legend(frameon=False)
    axes[0].set_ylabel("Richness readout")
    fig.savefig(outdir / "l1_objective_comparison.png", dpi=180)
    plt.close(fig)


def make_report(
    rows: list[dict[str, float | str]],
    diagnostics: list[dict[str, float | str]],
    outdir: Path,
    T: int,
    n_rollouts: int,
) -> None:
    def row_for(objective: str, sigma: float, E: float) -> dict[str, float | str]:
        return next(
            row
            for row in rows
            if row["objective"] == objective
            and abs(float(row["sigma"]) - sigma) < 1e-9
            and abs(float(row["E"]) - E) < 1e-9
        )

    lp_low_sigma = row_for("learning_progress", 0.0, 1.0)
    lp_high_sigma = row_for("learning_progress", 1.0, 1.0)
    lp_low_E = row_for("learning_progress", 0.5, 0.0)
    lp_high_E = row_for("learning_progress", 0.5, 1.0)

    combined_candidates = [row for row in rows if row["objective"] == "learning_progress"]
    best_combined = max(combined_candidates, key=lambda row: float(row["combined_flow_stock"]))

    lines = [
        "# L1 Computational Model: Environment-Shaped Exploration and Richness",
        "",
        "## What changed relative to the earlier bandit simulations",
        "",
        "This is a computational-level model. The agent does not learn an action policy online. Instead, the code computes the optimal finite-horizon policy exactly by backward induction over a discretized exposure state.",
        "",
        "The output is not a single trajectory. The main output is a comparative-statics surface: richness as a function of environment structure (`sigma`) and enrichment/security (`E`).",
        "",
        "## Model",
        "",
        f"- Horizon: T = {T}",
        f"- Monte Carlo rollouts per grid point: {n_rollouts}",
        "- Arms: K = 5 experiences.",
        "- Exposure state: h_k in {0, 1, 2, 3, 4}.",
        "- Confidence/routinization curve: c_k(h) = 1 - exp(-gamma_k h).",
        "- Novelty: 1 - c_k(h) = exp(-gamma_k h).",
        "- `sigma`: probability that exposure persists after choosing an arm. High sigma means structure/stationarity and durable routinization. Low sigma means drift/refresh.",
        "- `E`: enrichment/security. Low E penalizes risky/ambiguous arms.",
        "",
        "## Objective functions",
        "",
        "Two objectives were tested because the objective is the key modeling decision.",
        "",
        "1. `novelty`: reward equals current novelty, exp(-gamma_k h). Hard-to-learn arms remain novel, so the optimum tends to embrace them.",
        "2. `learning_progress`: reward equals delta confidence, c_k(h+1) - c_k(h). Hard-to-learn arms give less progress per step, so the optimum can drop them, especially under risk penalty.",
        "",
        "## Main diagnostics: learning_progress objective",
        "",
        "| Comparison | Flow richness | Stock richness | Combined flow/stock |",
        "|---|---:|---:|---:|",
        f"| Drift, high enrichment: sigma=0, E=1 | {float(lp_low_sigma['flow']):.3f} | {float(lp_low_sigma['stock']):.2f} | {float(lp_low_sigma['combined_flow_stock']):.3f} |",
        f"| Stationary, high enrichment: sigma=1, E=1 | {float(lp_high_sigma['flow']):.3f} | {float(lp_high_sigma['stock']):.2f} | {float(lp_high_sigma['combined_flow_stock']):.3f} |",
        f"| Mid-structure, deprivation: sigma=0.5, E=0 | {float(lp_low_E['flow']):.3f} | {float(lp_low_E['stock']):.2f} | {float(lp_low_E['combined_flow_stock']):.3f} |",
        f"| Mid-structure, enrichment: sigma=0.5, E=1 | {float(lp_high_E['flow']):.3f} | {float(lp_high_E['stock']):.2f} | {float(lp_high_E['combined_flow_stock']):.3f} |",
        "",
        f"The best combined flow/stock point under `learning_progress` is sigma={float(best_combined['sigma']):.2f}, E={float(best_combined['E']):.2f}, combined={float(best_combined['combined_flow_stock']):.3f}.",
        "",
        "## Extreme-environment policy diagnostics",
        "",
        "| Objective | Environment | Flow | Stock | Diagnostic action path |",
        "|---|---|---:|---:|---|",
    ]
    for diagnostic in diagnostics:
        lines.append(
            "| {objective} | {tag} | {flow:.3f} | {stock:.2f} | {actions} |".format(
                objective=diagnostic["objective"],
                tag=diagnostic["tag"],
                flow=float(diagnostic["flow"]),
                stock=float(diagnostic["stock"]),
                actions=diagnostic["actions"],
            )
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- P1, flow under drift: lower sigma refreshes exposure, so current novelty remains higher.",
            "- P2, learnability conversion loss: under `learning_progress`, hard-to-learn/risky arms are less attractive because they yield less confidence gain per step and can be priced out by low enrichment.",
            "- P3, deprivation retreat: lowering E reduces both flow and stock by penalizing risky/ambiguous arms.",
            "- P4, sweet spot: the combined readout is strongest where enrichment is high and structure is not simply maximal. The exact peak depends on the chosen objective.",
            "",
            "The objective-function comparison is central. If the objective is raw novelty, hard-to-learn arms stay attractive because they stay fresh. If the objective is learning progress, the same arms can be avoided because they do not convert exposure into confidence efficiently. This is the main modeling hinge.",
            "",
            "## Outputs",
            "",
            "- `results/l1_surface_results.csv`",
            "- `results/l1_policy_diagnostics.csv`",
            "- `results/l1_learning_progress_surfaces.png`",
            "- `results/l1_novelty_surfaces.png`",
            "- `results/l1_objective_comparison.png`",
        ]
    )
    (outdir / "l1_report.md").write_text("\n".join(lines) + "\n")


def run(outdir: Path, T: int, n_rollouts: int) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, float | str]] = []
    diagnostics: list[dict[str, float | str]] = []

    for objective in ("learning_progress", "novelty"):
        sigmas, Es, flow, stock, _avg_value, rows = sweep(
            objective=objective,
            T=T,
            n_rollouts=n_rollouts,
        )
        all_rows.extend(rows)
        plot_surface_pair(objective, sigmas, Es, flow, stock, outdir)

        for sigma, tag in ((1.0, "stationary"), (0.0, "drifting")):
            env = RichnessEnv(sigma=sigma, E=1.0, objective=objective)
            policy, idx = solve_optimal_policy(env, T=T)
            f, s, _v = measure_richness(env, policy, idx, T=T, n_rollouts=n_rollouts, seed=42)
            diagnostics.append(
                {
                    "objective": objective,
                    "sigma": sigma,
                    "E": 1.0,
                    "tag": tag,
                    "flow": f,
                    "stock": s,
                    "actions": " ".join(map(str, initial_state_action_sequence(env, policy, idx, T))),
                }
            )

    write_csv(outdir / "l1_surface_results.csv", all_rows)
    write_csv(outdir / "l1_policy_diagnostics.csv", diagnostics)
    plot_objective_comparison(all_rows, outdir)
    make_report(all_rows, diagnostics, outdir, T=T, n_rollouts=n_rollouts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=Path("results"))
    parser.add_argument("--T", type=int, default=12)
    parser.add_argument("--rollouts", type=int, default=1500)
    args = parser.parse_args()
    run(args.outdir, T=args.T, n_rollouts=args.rollouts)
    print(f"Wrote L1 outputs to {args.outdir.resolve()}")


if __name__ == "__main__":
    main()
