#!/usr/bin/env python3
"""Focused simulations for the final psychological richness story.

This script intentionally avoids a large parameter sweep. It runs three
targeted simulations:
1. Static versus flowing ecology.
2. Exploration-policy robustness.
3. Breadth versus depth.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

import matplotlib.pyplot as plt
import numpy as np

import simulate_richness as base


FOCUSED_SIMS = 200
DEPTH_SIMS = 500
SEED = 20260614


def write_csv(path: Path, rows: list[dict[str, str | float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, str | float]], keys: tuple[str, ...]) -> list[dict[str, str | float]]:
    groups: dict[tuple[str, ...], list[dict[str, str | float]]] = defaultdict(list)
    for row in rows:
        groups[tuple(str(row[key]) for key in keys)].append(row)
    metrics = [key for key in rows[0] if key not in {"simulation", *keys}]
    out: list[dict[str, str | float]] = []
    for key, group in sorted(groups.items()):
        item: dict[str, str | float] = dict(zip(keys, key))
        item["n_sims"] = len(group)
        for metric in metrics:
            values = [float(row[metric]) for row in group]
            item[f"{metric}_mean"] = mean(values)
            item[f"{metric}_sd"] = pstdev(values)
        out.append(item)
    return out


def history_metric(history: list[dict[str, float]], metric: str, start: int, end: int) -> float:
    return mean(row[metric] for row in history if start <= row["round"] <= end)


def run_stock_flow(rng: np.random.Generator, sims: int) -> tuple[list[dict[str, str | float]], list[dict[str, str | float]]]:
    condition = base.Condition(
        name="rich_affordable",
        ecology="rich",
        constraint="generous_budget_low_cost",
        example="campus / walkable mixed-use ecology",
        k=100,
        time_budget=0.80,
        money_budget=0.80,
        arm_cost_profile="low",
    )
    rows: list[dict[str, str | float]] = []
    time_rows: list[dict[str, str | float]] = []
    for arm_regime in ("fixed", "flowing"):
        for sim in range(sims):
            result, history = base.simulate_one(
                rng,
                condition,
                "thompson",
                record_history=True,
                arm_regime=arm_regime,
            )
            rows.append(
                {
                    "simulation": sim,
                    "arm_regime": arm_regime,
                    "experienced_richness": result["experienced_richness"],
                    "experienced_ecological": result["conversion_experienced_ecological"],
                    "early_discovery_rate": history_metric(history, "new_discovered", 1, 50),
                    "late_discovery_rate": history_metric(history, "new_discovered", 151, 200),
                    "novelty_decay": history_metric(history, "new_discovered", 1, 50)
                    - history_metric(history, "new_discovered", 151, 200),
                }
            )
            for item in history:
                time_rows.append(
                    {
                        "simulation": sim,
                        "arm_regime": arm_regime,
                        "round": item["round"],
                        "cumulative_unique": item["cumulative_unique"],
                        "new_discovered": item["new_discovered"],
                        "active_ecological_richness": item["active_ecological_richness"],
                    }
                )
    return rows, time_rows


def run_policy(rng: np.random.Generator, sims: int) -> list[dict[str, str | float]]:
    condition = base.Condition(
        name="rich_affordable_flowing",
        ecology="rich",
        constraint="generous_budget_low_cost",
        example="continuously renewing feasible ecology",
        k=100,
        time_budget=0.80,
        money_budget=0.80,
        arm_cost_profile="low",
    )
    policies = {
        "low_exploration": "egreedy_0.1",
        "adaptive_exploration": "thompson",
        "forced_exploration": "egreedy_0.5",
    }
    rows: list[dict[str, str | float]] = []
    for label, algorithm in policies.items():
        for sim in range(sims):
            result, history = base.simulate_one(
                rng,
                condition,
                algorithm,
                record_history=True,
                arm_regime="flowing",
            )
            rows.append(
                {
                    "simulation": sim,
                    "policy": label,
                    "algorithm": algorithm,
                    "experienced_richness": result["experienced_richness"],
                    "experienced_ecological": result["conversion_experienced_ecological"],
                    "early_discovery_rate": history_metric(history, "new_discovered", 1, 50),
                    "late_discovery_rate": history_metric(history, "new_discovered", 151, 200),
                    "top_arm_concentration": result["top_arm_concentration"],
                    "avg_reward": result["avg_reward"],
                }
            )
    return rows


def breadth_depth_event(arm_type: str, prior_visits: int) -> float:
    if arm_type == "surface":
        return 1.0 if prior_visits == 0 else 0.05
    if arm_type == "deep":
        return min(1.0, 0.12 + 0.10 * prior_visits)
    raise ValueError(arm_type)


def run_breadth_depth(rng: np.random.Generator, sims: int) -> list[dict[str, str | float]]:
    environments = {
        "breadth_city": {"k": 100, "surface_share": 0.80},
        "depth_town": {"k": 10, "surface_share": 0.20},
    }
    rows: list[dict[str, str | float]] = []
    for env, params in environments.items():
        k = int(params["k"])
        n_surface = int(round(k * float(params["surface_share"])))
        arm_types = np.array(["surface"] * n_surface + ["deep"] * (k - n_surface))
        for sim in range(sims):
            rng.shuffle(arm_types)
            condition = base.Condition(
                name=env,
                ecology=f"K={k}",
                constraint="all_feasible",
                example=env,
                k=k,
                time_budget=0.95,
                money_budget=0.95,
                arm_cost_profile="low",
            )
            alpha = np.ones(k)
            beta = np.ones(k)
            feasible = np.ones(k, dtype=bool)
            counts = np.zeros(k, dtype=np.int16)
            richness_events: list[float] = []
            new_events: list[float] = []
            for t in range(base.T_ROUNDS):
                arm = base.choose_arm(rng, "thompson", alpha, beta, feasible, t)
                prior = int(counts[arm])
                richness_events.append(breadth_depth_event(str(arm_types[arm]), prior))
                new_events.append(float(prior == 0))
                reward = int(rng.random() < base.TRUE_P)
                counts[arm] += 1
                alpha[arm] += reward
                beta[arm] += 1 - reward
            rows.append(
                {
                    "simulation": sim,
                    "environment": env,
                    "k": k,
                    "surface_share": float(params["surface_share"]),
                    "unique_arms": float(np.count_nonzero(counts)),
                    "unique_share": float(np.count_nonzero(counts) / k),
                    "cumulative_richness": float(sum(richness_events)),
                    "avg_richness_per_round": float(mean(richness_events)),
                    "early_richness_flow": mean(richness_events[:50]),
                    "late_richness_flow": mean(richness_events[150:200]),
                    "late_new_discovery_rate": mean(new_events[150:200]),
                    "deep_visit_share": float(
                        sum(counts[i] for i, arm_type in enumerate(arm_types) if arm_type == "deep")
                        / base.T_ROUNDS
                    ),
                }
            )
    return rows


def plot_focused(
    stock_flow_summary: list[dict[str, str | float]],
    policy_summary: list[dict[str, str | float]],
    breadth_depth_summary: list[dict[str, str | float]],
    outdir: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.5), constrained_layout=True)

    regimes = ["fixed", "flowing"]
    axes[0].bar(
        np.arange(2) - 0.18,
        [
            float(next(row for row in stock_flow_summary if row["arm_regime"] == regime)["experienced_richness_mean"])
            for regime in regimes
        ],
        width=0.36,
        label="final stock",
        color="#4C78A8",
    )
    axes[0].bar(
        np.arange(2) + 0.18,
        [
            float(next(row for row in stock_flow_summary if row["arm_regime"] == regime)["late_discovery_rate_mean"])
            for regime in regimes
        ],
        width=0.36,
        label="late flow",
        color="#E45756",
    )
    axes[0].set_xticks(np.arange(2))
    axes[0].set_xticklabels(regimes)
    axes[0].set_title("Static vs flowing ecology")
    axes[0].legend(frameon=False, fontsize=8)

    policies = ["low_exploration", "adaptive_exploration", "forced_exploration"]
    axes[1].bar(
        policies,
        [
            float(next(row for row in policy_summary if row["policy"] == policy)["late_discovery_rate_mean"])
            for policy in policies
        ],
        color=["#4C78A8", "#54A24B", "#F58518"],
    )
    axes[1].set_title("Exploration policy robustness")
    axes[1].set_ylabel("Late discovery rate")
    axes[1].tick_params(axis="x", rotation=20)

    envs = ["breadth_city", "depth_town"]
    axes[2].bar(
        np.arange(2) - 0.18,
        [
            float(next(row for row in breadth_depth_summary if row["environment"] == env)["unique_arms_mean"])
            for env in envs
        ],
        width=0.36,
        label="unique arms",
        color="#72B7B2",
    )
    axes[2].bar(
        np.arange(2) + 0.18,
        [
            float(next(row for row in breadth_depth_summary if row["environment"] == env)["late_richness_flow_mean"])
            for env in envs
        ],
        width=0.36,
        label="late richness flow",
        color="#B279A2",
    )
    axes[2].set_xticks(np.arange(2))
    axes[2].set_xticklabels(["breadth\ncity", "depth\ntown"])
    axes[2].set_title("Breadth vs depth")
    axes[2].legend(frameon=False, fontsize=8)

    for axis in axes:
        axis.grid(True, axis="y", alpha=0.25)
    fig.savefig(outdir / "focused_simulations.png", dpi=180)
    plt.close(fig)


def make_focused_report(
    stock_flow_summary: list[dict[str, str | float]],
    policy_summary: list[dict[str, str | float]],
    breadth_depth_summary: list[dict[str, str | float]],
    outdir: Path,
    sims: int,
    depth_sims: int,
) -> None:
    def sf(regime: str) -> dict[str, str | float]:
        return next(row for row in stock_flow_summary if row["arm_regime"] == regime)

    def pol(policy: str) -> dict[str, str | float]:
        return next(row for row in policy_summary if row["policy"] == policy)

    def bd(env: str) -> dict[str, str | float]:
        return next(row for row in breadth_depth_summary if row["environment"] == env)

    fixed = sf("fixed")
    flowing = sf("flowing")
    low = pol("low_exploration")
    adaptive = pol("adaptive_exploration")
    forced = pol("forced_exploration")
    breadth = bd("breadth_city")
    depth = bd("depth_town")

    lines = [
        "# Focused Simulation Report: From Ecological Opportunity to Experienced Richness",
        "",
        "## Scope",
        "",
        f"This focused report uses {sims} simulations for stock/flow and policy tests, plus {depth_sims} simulations for the breadth/depth extension. It deliberately avoids broad parameter search and keeps only three targeted simulations.",
        "",
        "## Simulation 1: Static vs Flowing Ecology",
        "",
        "| Ecology type | Final experienced richness | Late novelty rate | Novelty decay |",
        "|---|---:|---:|---:|",
        f"| static | {float(fixed['experienced_richness_mean']):.1f} | {float(fixed['late_discovery_rate_mean']):.3f} | {float(fixed['novelty_decay_mean']):.3f} |",
        f"| flowing | {float(flowing['experienced_richness_mean']):.1f} | {float(flowing['late_discovery_rate_mean']):.3f} | {float(flowing['novelty_decay_mean']):.3f} |",
        "",
        "Interpretation: static abundance mainly creates stock richness. Flowing ecology is more important for flow richness because feasible new options keep arriving.",
        "",
        "## Simulation 2: Exploration Policy Robustness",
        "",
        "| Policy | Algorithm | Final experienced richness | Late novelty rate | Top-arm concentration |",
        "|---|---|---:|---:|---:|",
        f"| low exploration | e-greedy 0.1 | {float(low['experienced_richness_mean']):.1f} | {float(low['late_discovery_rate_mean']):.3f} | {float(low['top_arm_concentration_mean']):.3f} |",
        f"| adaptive exploration | Thompson | {float(adaptive['experienced_richness_mean']):.1f} | {float(adaptive['late_discovery_rate_mean']):.3f} | {float(adaptive['top_arm_concentration_mean']):.3f} |",
        f"| forced exploration | e-greedy 0.5 | {float(forced['experienced_richness_mean']):.1f} | {float(forced['late_discovery_rate_mean']):.3f} | {float(forced['top_arm_concentration_mean']):.3f} |",
        "",
        "Interpretation: different learning policies produce different kinds of rich lives. Low exploration routinizes, forced exploration maximizes novelty, and Thompson is the most psychologically plausible middle case.",
        "",
        "## Simulation 3: Breadth vs Depth",
        "",
        "| Environment | K | Unique arms | Unique/K | Late new-discovery rate | Late richness flow | Cumulative richness |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| breadth city | {int(float(breadth['k']))} | {float(breadth['unique_arms_mean']):.1f} | {float(breadth['unique_share_mean']):.3f} | {float(breadth['late_new_discovery_rate_mean']):.3f} | {float(breadth['late_richness_flow_mean']):.3f} | {float(breadth['cumulative_richness_mean']):.1f} |",
        f"| depth town | {int(float(depth['k']))} | {float(depth['unique_arms_mean']):.1f} | {float(depth['unique_share_mean']):.3f} | {float(depth['late_new_discovery_rate_mean']):.3f} | {float(depth['late_richness_flow_mean']):.3f} | {float(depth['cumulative_richness_mean']):.1f} |",
        "",
        "Interpretation: if richness includes depth, small ecologies are not automatically poor. Repeated engagement with deep arms can sustain meaning even when new-arm discovery is low.",
        "",
        "## Four Conclusions",
        "",
        "1. Ecological richness does not automatically become experienced richness. Ecological opportunity is not the same as lived experience.",
        "2. The conversion pathway matters: ecological richness -> feasible richness -> experienced richness. Time, cost, and access can block conversion.",
        "3. Stock richness and flow richness are different. Stock richness is accumulated diversity; flow richness is ongoing novelty.",
        "4. Rich ecologies matter most when they generate continuing feasible novelty. The psychologically rich environment is not simply the place with the most options, but the place where new options keep becoming livable.",
        "",
        "## Suggested Title",
        "",
        "**From Ecological Opportunity to Experienced Richness**",
        "",
        "Subtitle: **Why more possible lives do not always produce richer lived lives**",
        "",
        "Alternative theory-forward title: **Stock and Flow in Psychological Richness**",
        "",
        "## Files",
        "",
        "- `results/focused_stock_flow.csv`",
        "- `results/focused_policy.csv`",
        "- `results/focused_breadth_depth.csv`",
        "- `results/focused_simulations.png`",
    ]
    (outdir / "focused_report.md").write_text("\n".join(lines) + "\n")


def run(outdir: Path, sims: int, depth_sims: int) -> None:
    rng = np.random.default_rng(SEED)
    stock_flow_rows, stock_flow_time_rows = run_stock_flow(rng, sims)
    policy_rows = run_policy(rng, sims)
    breadth_depth_rows = run_breadth_depth(rng, depth_sims)

    stock_flow_summary = summarize(stock_flow_rows, ("arm_regime",))
    policy_summary = summarize(policy_rows, ("policy", "algorithm"))
    breadth_depth_summary = summarize(breadth_depth_rows, ("environment", "k"))

    write_csv(outdir / "focused_stock_flow.csv", stock_flow_rows)
    write_csv(outdir / "focused_stock_flow_summary.csv", stock_flow_summary)
    write_csv(outdir / "focused_stock_flow_time.csv", stock_flow_time_rows)
    write_csv(outdir / "focused_policy.csv", policy_rows)
    write_csv(outdir / "focused_policy_summary.csv", policy_summary)
    write_csv(outdir / "focused_breadth_depth.csv", breadth_depth_rows)
    write_csv(outdir / "focused_breadth_depth_summary.csv", breadth_depth_summary)
    plot_focused(stock_flow_summary, policy_summary, breadth_depth_summary, outdir)
    make_focused_report(stock_flow_summary, policy_summary, breadth_depth_summary, outdir, sims, depth_sims)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="results", type=Path)
    parser.add_argument("--sims", default=FOCUSED_SIMS, type=int)
    parser.add_argument("--depth-sims", default=DEPTH_SIMS, type=int)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    run(args.outdir, args.sims, args.depth_sims)
    print(f"Wrote focused simulation outputs to {args.outdir.resolve()}")


if __name__ == "__main__":
    main()
