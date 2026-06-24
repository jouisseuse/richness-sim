#!/usr/bin/env python3
"""Ecological, feasible, and experienced richness simulation.

Core model:
- Every arm is an experience with identical Bernoulli reward p=0.9.
- Every arm also has a time cost and money cost.
- Agents can only choose arms that fit their time and money budgets.
- Beliefs are Beta-Bernoulli posteriors updated by Bayes' rule.

The key distinction is:
ecological richness = all existing options
feasible richness = options the agent can afford in time and money
experienced richness = options the agent actually chooses
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev

import matplotlib.pyplot as plt
import numpy as np


TRUE_P = 0.9
T_ROUNDS = 200
DEFAULT_SIMS = 200
DEFAULT_SENSITIVITY_SIMS = 50
SEED = 20260613
ALGORITHMS = ("egreedy_0.1", "egreedy_0.5", "egreedy_0.9", "thompson", "ucb")
TEMPORAL_ALGORITHMS = ("egreedy_0.1", "egreedy_0.5", "thompson", "ucb")
ARM_REGIMES = ("fixed", "flowing")
BUDGET_LEVELS = {
    "tight": 0.35,
    "moderate": 0.55,
    "generous": 0.80,
}
COST_PROFILES = ("low", "high")


@dataclass(frozen=True)
class Condition:
    name: str
    ecology: str
    constraint: str
    example: str
    k: int
    time_budget: float
    money_budget: float
    arm_cost_profile: str


CONDITIONS = (
    Condition(
        name="big_city",
        ecology="rich",
        constraint="low_time_high_cost",
        example="big city",
        k=100,
        time_budget=0.35,
        money_budget=0.35,
        arm_cost_profile="high",
    ),
    Condition(
        name="small_town",
        ecology="poor",
        constraint="high_time_low_cost",
        example="small town",
        k=10,
        time_budget=0.80,
        money_budget=0.80,
        arm_cost_profile="low",
    ),
    Condition(
        name="ideal_city",
        ecology="rich",
        constraint="high_time_low_cost",
        example="ideal city / campus",
        k=100,
        time_budget=0.80,
        money_budget=0.80,
        arm_cost_profile="low",
    ),
    Condition(
        name="constrained_rural",
        ecology="poor",
        constraint="low_time_high_cost",
        example="constrained rural / isolated work life",
        k=10,
        time_budget=0.35,
        money_budget=0.35,
        arm_cost_profile="high",
    ),
)

ECOLOGY_GRADIENT_CONDITIONS = tuple(
    Condition(
        name=f"ecology_k_{k}",
        ecology=f"K={k}",
        constraint="high_time_low_cost",
        example="ecology gradient",
        k=k,
        time_budget=0.80,
        money_budget=0.80,
        arm_cost_profile="low",
    )
    for k in (10, 25, 50, 100)
)


def entropy(counts: np.ndarray) -> float:
    nonzero = counts[counts > 0]
    if len(nonzero) == 0:
        return 0.0
    probs = nonzero / np.sum(nonzero)
    return float(-np.sum(probs * np.log(probs)))


def generate_arm_costs(
    rng: np.random.Generator,
    condition: Condition,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate time cost, money cost, and intrinsic novelty per arm."""
    if condition.arm_cost_profile == "low":
        base = rng.beta(2.0, 6.0, condition.k)
    elif condition.arm_cost_profile == "high":
        base = rng.beta(3.0, 3.0, condition.k)
    else:
        raise ValueError(f"Unknown arm cost profile: {condition.arm_cost_profile}")

    noise = rng.normal(0.0, 0.06, (2, condition.k))
    time_cost = np.clip(base + noise[0], 0.03, 0.98)
    money_cost = np.clip(base + noise[1], 0.03, 0.98)
    novelty = np.ones(condition.k, dtype=float)
    return time_cost, money_cost, novelty


def choose_arm(
    rng: np.random.Generator,
    algorithm: str,
    alpha: np.ndarray,
    beta: np.ndarray,
    feasible: np.ndarray,
    t: int,
) -> int:
    feasible_arms = np.flatnonzero(feasible)
    pulls = alpha[feasible_arms] + beta[feasible_arms] - 2
    untried = feasible_arms[pulls == 0]

    if algorithm.startswith("egreedy"):
        epsilon = float(algorithm.split("_")[1])
        if len(untried) and rng.random() < epsilon:
            return int(rng.choice(untried))
        posterior_mean = alpha[feasible_arms] / (alpha[feasible_arms] + beta[feasible_arms])
        best = np.max(posterior_mean)
        return int(rng.choice(feasible_arms[posterior_mean == best]))

    if algorithm == "thompson":
        samples = rng.beta(alpha[feasible_arms], beta[feasible_arms])
        best = np.max(samples)
        return int(rng.choice(feasible_arms[samples == best]))

    if algorithm == "ucb":
        if len(untried):
            return int(rng.choice(untried))
        posterior_mean = alpha[feasible_arms] / (alpha[feasible_arms] + beta[feasible_arms])
        bonus = np.sqrt(2.0 * math.log(max(t + 1, 2)) / pulls)
        score = posterior_mean + bonus
        best = np.max(score)
        return int(rng.choice(feasible_arms[score == best]))

    raise ValueError(f"Unknown algorithm: {algorithm}")


def active_arms_for_round(
    rng: np.random.Generator,
    k: int,
    t: int,
    arm_regime: str,
    active: np.ndarray,
) -> np.ndarray:
    if arm_regime == "fixed":
        active[:] = True
        return active

    if arm_regime == "flowing":
        if t == 0 and not np.any(active):
            active[: min(5, k)] = True
        remaining = np.flatnonzero(~active)
        if len(remaining):
            # Richer ecologies keep producing new possible experiences, but not
            # all at once. This makes ongoing novelty empirically visible.
            arrival_prob = min(0.85, 0.06 + k / 180.0)
            if rng.random() < arrival_prob:
                n_new = 1 + int(k >= 50 and rng.random() < 0.35)
                chosen = rng.choice(remaining, size=min(n_new, len(remaining)), replace=False)
                active[chosen] = True
        return active

    raise ValueError(f"Unknown arm regime: {arm_regime}")


def simulate_one(
    rng: np.random.Generator,
    condition: Condition,
    algorithm: str,
    record_history: bool = False,
    arm_regime: str = "fixed",
) -> tuple[dict[str, float], list[dict[str, float]]]:
    time_cost, money_cost, novelty = generate_arm_costs(rng, condition)
    feasible = (time_cost <= condition.time_budget) & (money_cost <= condition.money_budget)
    feasible_count = int(np.count_nonzero(feasible))

    # If the environment is extremely constrained, ensure the agent has at least
    # one affordably reachable option; otherwise the bandit cannot run.
    if feasible_count == 0:
        cheapest = int(np.argmin(time_cost + money_cost))
        feasible[cheapest] = True
        feasible_count = 1

    alpha = np.ones(condition.k, dtype=float)
    beta = np.ones(condition.k, dtype=float)
    choice_counts = np.zeros(condition.k, dtype=np.int16)
    choices = np.zeros(T_ROUNDS, dtype=np.int16)
    rewards = np.zeros(T_ROUNDS, dtype=np.int8)
    exploratory = 0
    history: list[dict[str, float]] = []
    active = np.zeros(condition.k, dtype=bool)

    for t in range(T_ROUNDS):
        active = active_arms_for_round(rng, condition.k, t, arm_regime, active)
        currently_feasible = feasible & active
        if not np.any(currently_feasible):
            currently_feasible[np.flatnonzero(feasible)[0]] = True
        arm = choose_arm(rng, algorithm, alpha, beta, currently_feasible, t)
        was_new = choice_counts[arm] == 0
        reward = int(rng.random() < TRUE_P)

        choices[t] = arm
        rewards[t] = reward
        choice_counts[arm] += 1
        exploratory += int(was_new)
        alpha[arm] += reward
        beta[arm] += 1 - reward
        novelty[arm] = 0.0

        if record_history:
            cumulative_unique = int(np.count_nonzero(choice_counts))
            active_count = int(np.count_nonzero(active))
            active_feasible_count = int(np.count_nonzero(currently_feasible))
            history.append(
                {
                    "round": float(t + 1),
                    "active_ecological_richness": float(active_count),
                    "active_feasible_richness": float(active_feasible_count),
                    "cumulative_unique": float(cumulative_unique),
                    "unique_share": cumulative_unique / condition.k,
                    "unique_over_active": cumulative_unique / active_count,
                    "new_discovered": float(was_new),
                    "feasible_richness": float(feasible_count),
                    "cumulative_unique_over_feasible": cumulative_unique / feasible_count,
                }
            )

    unique = int(np.count_nonzero(choice_counts))
    choice_entropy = entropy(choice_counts)
    feasible_entropy_max = math.log(feasible_count) if feasible_count > 1 else 0.0
    ecological_entropy_max = math.log(condition.k)
    switches = float(np.mean(choices[1:] != choices[:-1]))

    result = {
        "ecological_richness": float(condition.k),
        "feasible_richness": float(feasible_count),
        "experienced_richness": float(unique),
        "unexplored_options": float(condition.k - unique),
        "unaffordable_options": float(condition.k - feasible_count),
        "conversion_feasible_ecological": feasible_count / condition.k,
        "conversion_experienced_feasible": unique / feasible_count,
        "conversion_experienced_ecological": unique / condition.k,
        "choice_entropy": choice_entropy,
        "entropy_over_feasible_max": choice_entropy / feasible_entropy_max
        if feasible_entropy_max > 0
        else 0.0,
        "entropy_over_ecological_max": choice_entropy / ecological_entropy_max,
        "explore_prop": exploratory / T_ROUNDS,
        "switching_rate": switches,
        "top_arm_concentration": float(np.max(choice_counts) / T_ROUNDS),
        "avg_reward": float(np.mean(rewards)),
        "avg_feasible_time_cost": float(np.mean(time_cost[feasible])),
        "avg_feasible_money_cost": float(np.mean(money_cost[feasible])),
    }
    return result, history


def summarize(rows: list[dict[str, str | float]]) -> list[dict[str, str | float]]:
    groups: dict[tuple[str, str], list[dict[str, str | float]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["condition"]), str(row["algorithm"]))].append(row)

    metadata = {
        "simulation",
        "condition",
        "algorithm",
        "example",
        "ecology",
        "constraint",
        "k",
        "time_budget",
        "money_budget",
        "arm_cost_profile",
    }
    metrics = [key for key in rows[0] if key not in metadata]
    out: list[dict[str, str | float]] = []
    for (condition, algorithm), group in sorted(groups.items()):
        first = group[0]
        item: dict[str, str | float] = {
            "condition": condition,
            "algorithm": algorithm,
            "example": first["example"],
            "ecology": first["ecology"],
            "constraint": first["constraint"],
            "k": first["k"],
            "time_budget": first["time_budget"],
            "money_budget": first["money_budget"],
            "n_sims": len(group),
        }
        for metric in metrics:
            values = [float(row[metric]) for row in group]
            item[f"{metric}_mean"] = mean(values)
            item[f"{metric}_sd"] = pstdev(values)
        out.append(item)
    return out


def summarize_temporal(rows: list[dict[str, str | float]]) -> list[dict[str, str | float]]:
    groups: dict[tuple[str, str, str, int], list[dict[str, str | float]]] = defaultdict(list)
    for row in rows:
        groups[
            (
                str(row["condition"]),
                str(row["algorithm"]),
                str(row["arm_regime"]),
                int(float(row["round"])),
            )
        ].append(row)

    metadata = {"simulation", "condition", "algorithm", "arm_regime", "k", "round"}
    metrics = [key for key in rows[0] if key not in metadata]
    out: list[dict[str, str | float]] = []
    for (condition, algorithm, arm_regime, round_number), group in sorted(groups.items()):
        first = group[0]
        item: dict[str, str | float] = {
            "condition": condition,
            "algorithm": algorithm,
            "arm_regime": arm_regime,
            "k": first["k"],
            "round": round_number,
            "n_sims": len(group),
        }
        for metric in metrics:
            values = [float(row[metric]) for row in group]
            item[f"{metric}_mean"] = mean(values)
            item[f"{metric}_sd"] = pstdev(values)
        out.append(item)
    return out


def summarize_sensitivity(rows: list[dict[str, str | float]]) -> list[dict[str, str | float]]:
    group_keys = (
        "k",
        "budget_level",
        "cost_profile",
        "algorithm",
        "arm_regime",
    )
    groups: dict[tuple[str, ...], list[dict[str, str | float]]] = defaultdict(list)
    for row in rows:
        groups[tuple(str(row[key]) for key in group_keys)].append(row)

    metadata = {
        "simulation",
        "condition",
        "example",
        "ecology",
        "constraint",
        "time_budget",
        "money_budget",
        *group_keys,
    }
    metrics = [key for key in rows[0] if key not in metadata]
    out: list[dict[str, str | float]] = []
    for key, group in sorted(groups.items()):
        item: dict[str, str | float] = dict(zip(group_keys, key))
        item["n_sims"] = len(group)
        for metric in metrics:
            values = [float(row[metric]) for row in group]
            item[f"{metric}_mean"] = mean(values)
            item[f"{metric}_sd"] = pstdev(values)
        out.append(item)
    return out


def write_csv(path: Path, rows: list[dict[str, str | float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(summary: list[dict[str, str | float]], outdir: Path) -> None:
    rows = [row for row in summary if row["algorithm"] == "thompson"]
    order = ["big_city", "small_town", "ideal_city", "constrained_rural"]
    rows = sorted(rows, key=lambda row: order.index(str(row["condition"])))
    labels = [str(row["condition"]).replace("_", "\n") for row in rows]
    x = np.arange(len(rows))

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
    axes[0].bar(
        x - 0.2,
        [float(row["ecological_richness_mean"]) for row in rows],
        width=0.2,
        label="ecological",
        color="#4C78A8",
    )
    axes[0].bar(
        x,
        [float(row["feasible_richness_mean"]) for row in rows],
        width=0.2,
        label="feasible",
        color="#F58518",
    )
    axes[0].bar(
        x + 0.2,
        [float(row["experienced_richness_mean"]) for row in rows],
        width=0.2,
        label="experienced",
        color="#54A24B",
    )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("Number of arms")
    axes[0].set_title("Richness layers under Thompson sampling")
    axes[0].legend(frameon=False)
    axes[0].grid(True, axis="y", alpha=0.25)

    axes[1].bar(
        x - 0.2,
        [float(row["conversion_feasible_ecological_mean"]) for row in rows],
        width=0.2,
        label="feasible/ecological",
        color="#B279A2",
    )
    axes[1].bar(
        x,
        [float(row["conversion_experienced_feasible_mean"]) for row in rows],
        width=0.2,
        label="experienced/feasible",
        color="#E45756",
    )
    axes[1].bar(
        x + 0.2,
        [float(row["conversion_experienced_ecological_mean"]) for row in rows],
        width=0.2,
        label="experienced/ecological",
        color="#72B7B2",
    )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("Conversion rate")
    axes[1].set_title("Conversion efficiencies")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].grid(True, axis="y", alpha=0.25)

    fig.savefig(outdir / "richness_layers_thompson.png", dpi=180)
    plt.close(fig)


def plot_temporal(temporal_summary: list[dict[str, str | float]], outdir: Path) -> None:
    order = ["ecology_k_10", "ecology_k_25", "ecology_k_50", "ecology_k_100"]
    colors = {
        "ecology_k_10": "#4C78A8",
        "ecology_k_25": "#F58518",
        "ecology_k_50": "#54A24B",
        "ecology_k_100": "#B279A2",
    }
    labels = {
        "ecology_k_10": "K=10",
        "ecology_k_25": "K=25",
        "ecology_k_50": "K=50",
        "ecology_k_100": "K=100",
    }

    by_condition = {
        condition: [
            row
            for row in temporal_summary
            if row["condition"] == condition
            and row["algorithm"] == "thompson"
            and row["arm_regime"] == "fixed"
        ]
        for condition in order
    }

    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    for condition in order:
        rows = sorted(by_condition[condition], key=lambda row: int(row["round"]))
        ax.plot(
            [int(row["round"]) for row in rows],
            [float(row["cumulative_unique_mean"]) for row in rows],
            label=labels[condition],
            color=colors[condition],
        )
    ax.set_xlabel("Round")
    ax.set_ylabel("Cumulative unique arms")
    ax.set_title("Figure 1. Cumulative experienced richness")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(outdir / "figure1_cumulative_unique.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    window = 10
    kernel = np.ones(window) / window
    for condition in order:
        rows = sorted(by_condition[condition], key=lambda row: int(row["round"]))
        rounds = np.array([int(row["round"]) for row in rows])
        discoveries = np.array([float(row["new_discovered_mean"]) for row in rows])
        smooth = np.convolve(discoveries, kernel, mode="same")
        ax.plot(rounds, smooth, label=labels[condition], color=colors[condition])
    ax.set_xlabel("Round")
    ax.set_ylabel("New arms discovered per round")
    ax.set_title("Figure 2. Ongoing novelty rate")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(outdir / "figure2_new_discoveries.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    for condition in order:
        rows = sorted(by_condition[condition], key=lambda row: int(row["round"]))
        ax.plot(
            [int(row["round"]) for row in rows],
            [float(row["unique_share_mean"]) for row in rows],
            label=labels[condition],
            color=colors[condition],
        )
    ax.set_xlabel("Round")
    ax.set_ylabel("Cumulative unique arms / K")
    ax.set_ylim(0, 1.05)
    ax.set_title("Figure 3. Relative experienced richness")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(outdir / "figure3_unique_share.png", dpi=180)
    plt.close(fig)

    algorithm_colors = {
        "egreedy_0.1": "#4C78A8",
        "egreedy_0.5": "#F58518",
        "thompson": "#54A24B",
        "ucb": "#B279A2",
    }
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
    for algorithm in TEMPORAL_ALGORITHMS:
        rows = sorted(
            [
                row
                for row in temporal_summary
                if row["condition"] == "ecology_k_100"
                and row["algorithm"] == algorithm
                and row["arm_regime"] == "fixed"
            ],
            key=lambda row: int(row["round"]),
        )
        axes[0].plot(
            [int(row["round"]) for row in rows],
            [float(row["cumulative_unique_mean"]) for row in rows],
            label=algorithm,
            color=algorithm_colors[algorithm],
        )
        discoveries = np.array([float(row["new_discovered_mean"]) for row in rows])
        smooth = np.convolve(discoveries, kernel, mode="same")
        axes[1].plot(
            [int(row["round"]) for row in rows],
            smooth,
            label=algorithm,
            color=algorithm_colors[algorithm],
        )
    axes[0].set_xlabel("Round")
    axes[0].set_ylabel("Cumulative unique arms")
    axes[0].set_title("Figure 4a. Algorithm comparison, K=100")
    axes[1].set_xlabel("Round")
    axes[1].set_ylabel("New arms discovered per round")
    axes[1].set_title("Figure 4b. Algorithm novelty rate, K=100")
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
    fig.savefig(outdir / "figure4_algorithm_comparison.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
    regime_colors = {"fixed": "#4C78A8", "flowing": "#E45756"}
    for arm_regime in ARM_REGIMES:
        rows = sorted(
            [
                row
                for row in temporal_summary
                if row["condition"] == "ecology_k_100"
                and row["algorithm"] == "thompson"
                and row["arm_regime"] == arm_regime
            ],
            key=lambda row: int(row["round"]),
        )
        rounds = [int(row["round"]) for row in rows]
        axes[0].plot(
            rounds,
            [float(row["active_ecological_richness_mean"]) for row in rows],
            label=f"{arm_regime}: active ecology",
            color=regime_colors[arm_regime],
            linestyle="--",
        )
        axes[0].plot(
            rounds,
            [float(row["cumulative_unique_mean"]) for row in rows],
            label=f"{arm_regime}: experienced",
            color=regime_colors[arm_regime],
        )
        discoveries = np.array([float(row["new_discovered_mean"]) for row in rows])
        smooth = np.convolve(discoveries, kernel, mode="same")
        axes[1].plot(rounds, smooth, label=arm_regime, color=regime_colors[arm_regime])
    axes[0].set_xlabel("Round")
    axes[0].set_ylabel("Arms")
    axes[0].set_title("Figure 5a. Fixed vs flowing arms")
    axes[1].set_xlabel("Round")
    axes[1].set_ylabel("New arms discovered per round")
    axes[1].set_title("Figure 5b. Flowing arms and ongoing novelty")
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
    fig.savefig(outdir / "figure5_flowing_arms.png", dpi=180)
    plt.close(fig)


def plot_sensitivity(sensitivity_summary: list[dict[str, str | float]], outdir: Path) -> None:
    algorithms = list(TEMPORAL_ALGORITHMS)
    budgets = ["tight", "moderate", "generous"]
    cost_profiles = ["low", "high"]
    colors = {
        "egreedy_0.1": "#4C78A8",
        "egreedy_0.5": "#F58518",
        "thompson": "#54A24B",
        "ucb": "#B279A2",
    }

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True, sharey=True)
    for axis, cost_profile in zip(axes, cost_profiles):
        x = np.arange(len(budgets))
        width = 0.18
        for idx, algorithm in enumerate(algorithms):
            values = []
            for budget in budgets:
                row = next(
                    row
                    for row in sensitivity_summary
                    if row["k"] == "100"
                    and row["budget_level"] == budget
                    and row["cost_profile"] == cost_profile
                    and row["algorithm"] == algorithm
                    and row["arm_regime"] == "flowing"
                )
                values.append(float(row["late_discovery_rate_mean"]))
            axis.bar(
                x + (idx - 1.5) * width,
                values,
                width=width,
                label=algorithm,
                color=colors[algorithm],
            )
        axis.set_title(f"K=100, flowing arms, {cost_profile} costs")
        axis.set_xticks(x)
        axis.set_xticklabels(budgets)
        axis.set_xlabel("Budget")
        axis.grid(True, axis="y", alpha=0.25)
    axes[0].set_ylabel("Late discovery rate, rounds 151-200")
    axes[1].legend(frameon=False, fontsize=8)
    fig.savefig(outdir / "figure6_sensitivity_late_novelty.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
    markers = {"fixed": "o", "flowing": "s"}
    for algorithm in algorithms:
        for arm_regime in ARM_REGIMES:
            rows = [
                row
                for row in sensitivity_summary
                if row["algorithm"] == algorithm and row["arm_regime"] == arm_regime
            ]
            ax.scatter(
                [float(row["conversion_experienced_ecological_mean"]) for row in rows],
                [float(row["late_discovery_rate_mean"]) for row in rows],
                label=f"{algorithm}, {arm_regime}",
                alpha=0.72,
                s=34,
                color=colors[algorithm],
                marker=markers[arm_regime],
                edgecolors="none",
            )
    ax.set_xlabel("Experienced / ecological conversion")
    ax.set_ylabel("Late discovery rate")
    ax.set_title("Figure 7. Total conversion vs ongoing novelty")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.savefig(outdir / "figure7_conversion_vs_late_novelty.png", dpi=180)
    plt.close(fig)


def mean_temporal_value(
    temporal_summary: list[dict[str, str | float]],
    condition: str,
    metric: str,
    start_round: int,
    end_round: int,
    algorithm: str = "thompson",
    arm_regime: str = "fixed",
) -> float:
    rows = [
        row
        for row in temporal_summary
        if row["condition"] == condition
        and row["algorithm"] == algorithm
        and row["arm_regime"] == arm_regime
        and start_round <= int(row["round"]) <= end_round
    ]
    return mean(float(row[f"{metric}_mean"]) for row in rows)


def temporal_value_at(
    temporal_summary: list[dict[str, str | float]],
    condition: str,
    metric: str,
    round_number: int,
    algorithm: str = "thompson",
    arm_regime: str = "fixed",
) -> float:
    row = next(
        row
        for row in temporal_summary
        if row["condition"] == condition
        and row["algorithm"] == algorithm
        and row["arm_regime"] == arm_regime
        and int(row["round"]) == round_number
    )
    return float(row[f"{metric}_mean"])


def add_history_summaries(
    result: dict[str, float],
    history: list[dict[str, float]],
) -> dict[str, float]:
    early = [row["new_discovered"] for row in history if 1 <= row["round"] <= 50]
    middle = [row["new_discovered"] for row in history if 76 <= row["round"] <= 125]
    late = [row["new_discovered"] for row in history if 151 <= row["round"] <= 200]
    result = dict(result)
    result["early_discovery_rate"] = mean(early)
    result["middle_discovery_rate"] = mean(middle)
    result["late_discovery_rate"] = mean(late)
    result["novelty_decay"] = result["early_discovery_rate"] - result["late_discovery_rate"]
    return result


def make_report(
    summary: list[dict[str, str | float]],
    temporal_summary: list[dict[str, str | float]],
    sensitivity_summary: list[dict[str, str | float]],
    outdir: Path,
    sims: int,
    sensitivity_sims: int,
) -> None:
    def get(condition: str, algorithm: str) -> dict[str, str | float]:
        return next(
            row
            for row in summary
            if row["condition"] == condition and row["algorithm"] == algorithm
        )

    focal_algorithm = "thompson"
    focal_rows = [get(condition, focal_algorithm) for condition in (
        "big_city",
        "small_town",
        "ideal_city",
        "constrained_rural",
    )]
    big_city = get("big_city", focal_algorithm)
    small_town = get("small_town", focal_algorithm)
    ideal_city = get("ideal_city", focal_algorithm)
    temporal_conditions = ("ecology_k_10", "ecology_k_25", "ecology_k_50", "ecology_k_100")

    lines = [
        "# Ecological, Feasible, and Experienced Richness Report",
        "",
        "## Model",
        "",
        f"- Ground truth reward: every arm is Bernoulli(p={TRUE_P}).",
        "- Prior belief for every arm: Beta(1, 1).",
        "- Bayesian update: success increments alpha; failure increments beta.",
        "- Algorithms: epsilon-greedy with epsilon = 0.1, 0.5, 0.9; Thompson sampling; UCB.",
        f"- Single-agent baseline; rounds = {T_ROUNDS}; simulations per condition = {sims}.",
        "- Each arm has time cost, money cost, and novelty. Novelty is initially 1 and becomes 0 after the arm is tried.",
        "- Agents can only choose arms within both their time budget and money budget.",
        "",
        "## Conditions",
        "",
        "| Condition | Ecology | Time/cost | Example | K | Time budget | Money budget | Arm costs |",
        "|---|---|---|---|---:|---:|---:|---|",
    ]
    for condition in CONDITIONS:
        lines.append(
            f"| {condition.name} | {condition.ecology} | {condition.constraint} | {condition.example} | {condition.k} | {condition.time_budget:.2f} | {condition.money_budget:.2f} | {condition.arm_cost_profile} |"
        )

    lines.extend(
        [
            "",
            f"## Main Results: {focal_algorithm}",
            "",
            "| Condition | Ecological | Feasible | Experienced | Unaffordable | Unexplored | Feasible/Ecological | Experienced/Feasible | Experienced/Ecological | Entropy | Avg reward |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in focal_rows:
        lines.append(
            "| {condition} | {eco:.0f} | {feasible:.1f} | {experienced:.1f} | {unaffordable:.1f} | {unexplored:.1f} | {c1:.3f} | {c2:.3f} | {c3:.3f} | {entropy:.2f} | {reward:.3f} |".format(
                condition=row["condition"],
                eco=float(row["ecological_richness_mean"]),
                feasible=float(row["feasible_richness_mean"]),
                experienced=float(row["experienced_richness_mean"]),
                unaffordable=float(row["unaffordable_options_mean"]),
                unexplored=float(row["unexplored_options_mean"]),
                c1=float(row["conversion_feasible_ecological_mean"]),
                c2=float(row["conversion_experienced_feasible_mean"]),
                c3=float(row["conversion_experienced_ecological_mean"]),
                entropy=float(row["choice_entropy_mean"]),
                reward=float(row["avg_reward_mean"]),
            )
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- Big city has the highest ecological richness among realistic conditions (K={int(float(big_city['ecological_richness_mean']))}), but only {float(big_city['feasible_richness_mean']):.1f} options were feasible on average under low time and high cost.",
            f"- Small town has lower ecological richness (K={int(float(small_town['ecological_richness_mean']))}), but its feasible/ecological conversion was {float(small_town['conversion_feasible_ecological_mean']):.1%}, versus {float(big_city['conversion_feasible_ecological_mean']):.1%} in big city.",
            f"- Ideal city/campus is the richest lived condition: many options plus enough time and affordable cost produced {float(ideal_city['experienced_richness_mean']):.1f} experienced arms on average.",
            f"- Big city still produced more absolute experienced richness than small town in this calibration ({float(big_city['experienced_richness_mean']):.1f} vs {float(small_town['experienced_richness_mean']):.1f}), but it also produced many more unrealized possibilities: {float(big_city['unaffordable_options_mean']):.1f} unaffordable and {float(big_city['unexplored_options_mean']):.1f} unexplored options.",
            "- The central pattern is not simply rich ecology -> rich experience. The conversion pathway matters: ecological richness -> feasible richness -> experienced richness.",
            "",
            "One-sentence takeaway: rich ecology may increase imagined possibilities more than lived experiences.",
            "",
            "## Temporal Richness Analysis",
            "",
            "This sub-experiment holds time and money budgets generous and affordable, then varies ecological K = 10, 25, 50, 100 under Thompson sampling. It asks whether richer ecology changes the timing of novelty, not just the final Round 200 total.",
            "",
            "| K | Unique at round 20 | Unique at round 100 | Unique at round 200 | Unique/K at round 200 | Discovery rate rounds 1-50 | Discovery rate rounds 151-200 |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for condition in temporal_conditions:
        k = int(condition.split("_")[-1])
        lines.append(
            "| {k} | {u20:.1f} | {u100:.1f} | {u200:.1f} | {share200:.3f} | {early:.3f} | {late:.3f} |".format(
                k=k,
                u20=temporal_value_at(temporal_summary, condition, "cumulative_unique", 20),
                u100=temporal_value_at(temporal_summary, condition, "cumulative_unique", 100),
                u200=temporal_value_at(temporal_summary, condition, "cumulative_unique", 200),
                share200=temporal_value_at(temporal_summary, condition, "unique_share", 200),
                early=mean_temporal_value(temporal_summary, condition, "new_discovered", 1, 50),
                late=mean_temporal_value(temporal_summary, condition, "new_discovered", 151, 200),
            )
        )

    k10_share = temporal_value_at(temporal_summary, "ecology_k_10", "unique_share", 200)
    k100_share = temporal_value_at(temporal_summary, "ecology_k_100", "unique_share", 200)
    k100_early = mean_temporal_value(temporal_summary, "ecology_k_100", "new_discovered", 1, 50)
    k100_late = mean_temporal_value(temporal_summary, "ecology_k_100", "new_discovered", 151, 200)
    flowing_late = mean_temporal_value(
        temporal_summary,
        "ecology_k_100",
        "new_discovered",
        151,
        200,
        algorithm="thompson",
        arm_regime="flowing",
    )
    fixed_late = mean_temporal_value(
        temporal_summary,
        "ecology_k_100",
        "new_discovered",
        151,
        200,
        algorithm="thompson",
        arm_regime="fixed",
    )

    algorithm_rows = []
    for algorithm in TEMPORAL_ALGORITHMS:
        algorithm_rows.append(
            {
                "algorithm": algorithm,
                "u200": temporal_value_at(
                    temporal_summary,
                    "ecology_k_100",
                    "cumulative_unique",
                    200,
                    algorithm=algorithm,
                    arm_regime="fixed",
                ),
                "late": mean_temporal_value(
                    temporal_summary,
                    "ecology_k_100",
                    "new_discovered",
                    151,
                    200,
                    algorithm=algorithm,
                    arm_regime="fixed",
                ),
            }
        )

    top_late = sorted(
        sensitivity_summary,
        key=lambda row: float(row["late_discovery_rate_mean"]),
        reverse=True,
    )[:8]
    top_conversion = sorted(
        sensitivity_summary,
        key=lambda row: float(row["conversion_experienced_ecological_mean"]),
        reverse=True,
    )[:8]
    thompson_flowing_generous = next(
        row
        for row in sensitivity_summary
        if row["k"] == "100"
        and row["budget_level"] == "generous"
        and row["cost_profile"] == "low"
        and row["algorithm"] == "thompson"
        and row["arm_regime"] == "flowing"
    )
    ucb_flowing_generous = next(
        row
        for row in sensitivity_summary
        if row["k"] == "100"
        and row["budget_level"] == "generous"
        and row["cost_profile"] == "low"
        and row["algorithm"] == "ucb"
        and row["arm_regime"] == "flowing"
    )
    egreedy_flowing_generous = next(
        row
        for row in sensitivity_summary
        if row["k"] == "100"
        and row["budget_level"] == "generous"
        and row["cost_profile"] == "low"
        and row["algorithm"] == "egreedy_0.5"
        and row["arm_regime"] == "flowing"
    )

    lines.extend(
        [
            "",
            f"- Figure 1 tests cumulative experienced richness. Larger ecologies maintain a higher absolute ceiling, so K=100 keeps accumulating more unique experiences than K=10.",
            f"- Figure 2 is the closest proxy for ongoing psychological richness: in K=100, the new-discovery rate falls from {k100_early:.3f} in rounds 1-50 to {k100_late:.3f} in rounds 151-200, showing routine formation even in rich ecology.",
            f"- Figure 3 shows the absolute/relative split: by round 200, K=10 reaches {k10_share:.1%} of its ecology, while K=100 reaches {k100_share:.1%}. Rich ecology creates more total novelty, but a smaller fraction of possible life is actually lived.",
            "",
            "## Algorithm And Flowing-Arm Checks",
            "",
            "| Algorithm | K=100 fixed: unique at round 200 | K=100 fixed: late discovery rate |",
            "|---|---:|---:|",
        ]
    )
    for row in algorithm_rows:
        lines.append(
            f"| {row['algorithm']} | {row['u200']:.1f} | {row['late']:.3f} |"
        )

    lines.extend(
        [
            "",
            f"- In K=100 under Thompson sampling, flowing arms keep late novelty higher than fixed arms ({flowing_late:.3f} vs {fixed_late:.3f}). This captures environments where new possibilities keep appearing rather than all existing from the start.",
            "- Figure 4 compares algorithms. Figure 5 compares fixed versus flowing arms.",
            "",
            "## Sensitivity Sweep",
            "",
            f"The broader sweep crosses K = 10, 25, 50, 100; budget = tight, moderate, generous; cost profile = low/high; algorithm = e-greedy 0.1, e-greedy 0.5, Thompson, UCB; and arm regime = fixed/flowing. Each cell uses {sensitivity_sims} simulations.",
            "",
            "Top conditions for sustained late novelty:",
            "",
            "| Rank | K | Budget | Cost | Algorithm | Arms | Experienced/Ecological | Late discovery | Novelty decay |",
            "|---:|---:|---|---|---|---|---:|---:|---:|",
        ]
    )
    for rank, row in enumerate(top_late, start=1):
        lines.append(
            "| {rank} | {k} | {budget} | {cost} | {algorithm} | {regime} | {conv:.3f} | {late:.3f} | {decay:.3f} |".format(
                rank=rank,
                k=row["k"],
                budget=row["budget_level"],
                cost=row["cost_profile"],
                algorithm=row["algorithm"],
                regime=row["arm_regime"],
                conv=float(row["conversion_experienced_ecological_mean"]),
                late=float(row["late_discovery_rate_mean"]),
                decay=float(row["novelty_decay_mean"]),
            )
        )

    lines.extend(
        [
            "",
            "Top conditions for total ecological conversion:",
            "",
            "| Rank | K | Budget | Cost | Algorithm | Arms | Experienced/Ecological | Late discovery | Feasible/Ecological |",
            "|---:|---:|---|---|---|---|---:|---:|---:|",
        ]
    )
    for rank, row in enumerate(top_conversion, start=1):
        lines.append(
            "| {rank} | {k} | {budget} | {cost} | {algorithm} | {regime} | {conv:.3f} | {late:.3f} | {feasible:.3f} |".format(
                rank=rank,
                k=row["k"],
                budget=row["budget_level"],
                cost=row["cost_profile"],
                algorithm=row["algorithm"],
                regime=row["arm_regime"],
                conv=float(row["conversion_experienced_ecological_mean"]),
                late=float(row["late_discovery_rate_mean"]),
                feasible=float(row["conversion_feasible_ecological_mean"]),
            )
        )

    lines.extend(
        [
            "",
            "## Best Interpretation",
            "",
            "- The most theoretically useful distinction is not high versus low final richness, but **stock novelty versus flow novelty**.",
            f"- UCB maximizes stock conversion in rich, affordable environments: for K=100/generous/low-cost/flowing arms, it reaches {float(ucb_flowing_generous['conversion_experienced_ecological_mean']):.1%} experienced/ecological conversion, but its late discovery rate is {float(ucb_flowing_generous['late_discovery_rate_mean']):.3f}. In other words, it exhausts the space and then novelty stops.",
            f"- E-greedy 0.5 keeps the strongest late novelty in the same environment ({float(egreedy_flowing_generous['late_discovery_rate_mean']):.3f}), but partly because it imposes constant random exploration. This is useful as an upper-bound exploration regime, not necessarily the most psychologically realistic learner.",
            f"- Thompson sampling is the best middle model for the theory: it is Bayesian, reward-sensitive, and still produces ongoing novelty when arms flow into the ecology. In K=100/generous/low-cost/flowing arms, it reaches {float(thompson_flowing_generous['conversion_experienced_ecological_mean']):.1%} total conversion and maintains late novelty at {float(thompson_flowing_generous['late_discovery_rate_mean']):.3f}.",
            "- The interpretation I would foreground is: rich ecologies matter most when they maintain a **continuing arrival process** of feasible options. Static abundance quickly becomes a stock of known possibilities; flowing abundance better captures eventfulness.",
            "",
            "",
            "## Output Files",
            "",
            "- `results/raw_simulations.csv`: one row per simulation run.",
            "- `results/summary_results.csv`: aggregated means and standard deviations.",
            "- `results/richness_layers_thompson.png`: ecological, feasible, and experienced richness visualization.",
            "- `results/temporal_raw.csv`: one row per temporal simulation round.",
            "- `results/temporal_summary.csv`: average temporal trajectories by K.",
            "- `results/figure1_cumulative_unique.png`: cumulative unique arms over time.",
            "- `results/figure2_new_discoveries.png`: new arms discovered per round over time.",
            "- `results/figure3_unique_share.png`: cumulative unique arms divided by K over time.",
            "- `results/figure4_algorithm_comparison.png`: algorithm comparison for K=100.",
            "- `results/figure5_flowing_arms.png`: fixed versus flowing arms comparison.",
            "- `results/figure6_sensitivity_late_novelty.png`: late novelty across budget/cost/algorithm settings.",
            "- `results/figure7_conversion_vs_late_novelty.png`: tradeoff between total conversion and ongoing novelty.",
            "- `results/sensitivity_raw.csv`: broad parameter sweep raw results.",
            "- `results/sensitivity_summary.csv`: broad parameter sweep summary.",
        ]
    )
    (outdir / "report.md").write_text("\n".join(lines) + "\n")


def run(outdir: Path, sims: int, sensitivity_sims: int) -> None:
    rng = np.random.default_rng(SEED)
    rows: list[dict[str, str | float]] = []
    for condition in CONDITIONS:
        for algorithm in ALGORITHMS:
            for sim in range(sims):
                result, _ = simulate_one(rng, condition, algorithm)
                row: dict[str, str | float] = {
                    "simulation": sim,
                    "condition": condition.name,
                    "algorithm": algorithm,
                    "example": condition.example,
                    "ecology": condition.ecology,
                    "constraint": condition.constraint,
                    "k": condition.k,
                    "time_budget": condition.time_budget,
                    "money_budget": condition.money_budget,
                    "arm_cost_profile": condition.arm_cost_profile,
                }
                row.update(result)
                rows.append(row)

    temporal_rows: list[dict[str, str | float]] = []
    for condition in ECOLOGY_GRADIENT_CONDITIONS:
        for algorithm in TEMPORAL_ALGORITHMS:
            for arm_regime in ARM_REGIMES:
                for sim in range(sims):
                    _, history = simulate_one(
                        rng,
                        condition,
                        algorithm,
                        record_history=True,
                        arm_regime=arm_regime,
                    )
                    for item in history:
                        temporal_rows.append(
                            {
                                "simulation": sim,
                                "condition": condition.name,
                                "algorithm": algorithm,
                                "arm_regime": arm_regime,
                                "k": condition.k,
                                **item,
                            }
                        )

    sensitivity_rows: list[dict[str, str | float]] = []
    for k in (10, 25, 50, 100):
        for budget_level, budget in BUDGET_LEVELS.items():
            for cost_profile in COST_PROFILES:
                condition = Condition(
                    name=f"sensitivity_k_{k}_{budget_level}_{cost_profile}",
                    ecology=f"K={k}",
                    constraint=f"{budget_level}_budget_{cost_profile}_cost",
                    example="sensitivity grid",
                    k=k,
                    time_budget=budget,
                    money_budget=budget,
                    arm_cost_profile=cost_profile,
                )
                for algorithm in TEMPORAL_ALGORITHMS:
                    for arm_regime in ARM_REGIMES:
                        for sim in range(sensitivity_sims):
                            result, history = simulate_one(
                                rng,
                                condition,
                                algorithm,
                                record_history=True,
                                arm_regime=arm_regime,
                            )
                            result = add_history_summaries(result, history)
                            sensitivity_rows.append(
                                {
                                    "simulation": sim,
                                    "condition": condition.name,
                                    "example": condition.example,
                                    "ecology": condition.ecology,
                                    "constraint": condition.constraint,
                                    "k": k,
                                    "budget_level": budget_level,
                                    "cost_profile": cost_profile,
                                    "time_budget": budget,
                                    "money_budget": budget,
                                    "algorithm": algorithm,
                                    "arm_regime": arm_regime,
                                    **result,
                                }
                            )

    summary = summarize(rows)
    temporal_summary = summarize_temporal(temporal_rows)
    sensitivity_summary = summarize_sensitivity(sensitivity_rows)
    write_csv(outdir / "raw_simulations.csv", rows)
    write_csv(outdir / "summary_results.csv", summary)
    write_csv(outdir / "temporal_raw.csv", temporal_rows)
    write_csv(outdir / "temporal_summary.csv", temporal_summary)
    write_csv(outdir / "sensitivity_raw.csv", sensitivity_rows)
    write_csv(outdir / "sensitivity_summary.csv", sensitivity_summary)
    plot_summary(summary, outdir)
    plot_temporal(temporal_summary, outdir)
    plot_sensitivity(sensitivity_summary, outdir)
    make_report(summary, temporal_summary, sensitivity_summary, outdir, sims, sensitivity_sims)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="results", type=Path)
    parser.add_argument("--sims", default=DEFAULT_SIMS, type=int)
    parser.add_argument("--sensitivity-sims", default=DEFAULT_SENSITIVITY_SIMS, type=int)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    run(args.outdir, args.sims, args.sensitivity_sims)
    print(f"Wrote simulation outputs to {args.outdir.resolve()}")


if __name__ == "__main__":
    main()
