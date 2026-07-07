#!/usr/bin/env python3
"""L1 model for new-campus / new-city life exploration.

Scenario:
You just moved to a new city or started at a new campus. Each week, you choose
one activity: cafes, museums, walking routes, clubs, research talks,
volunteering, sports, language exchange, community events, workshops, or
familiar routines.

This is a computational-level model:
- the agent does not learn a policy online;
- each choice maximizes a marginal future-useful-richness score;
- the main output is comparative statics over ecology and resource states.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev

import matplotlib.pyplot as plt
import numpy as np


SEED = 20260707
T_WEEKS = 52
EPS = 0.10
RHO_VALUES = (0.0, 0.25, 0.50, 0.75, 1.0)
GAMMA_LEVELS = {
    "low": 0.18,
    "medium": 0.45,
    "high": 0.90,
}
RESOURCE_STATES = {
    "enriched": {"cost_weight": 0.005, "risk_weight": 0.005},
    "depleted": {"cost_weight": 0.035, "risk_weight": 0.035},
}


@dataclass
class ActivitySet:
    gamma: np.ndarray
    q: np.ndarray
    cost: np.ndarray
    risk: np.ndarray
    h: np.ndarray
    active: np.ndarray

    @property
    def n_active(self) -> int:
        return int(np.count_nonzero(self.active))


def future_relevance(exposures: np.ndarray, active: np.ndarray, rho: float) -> np.ndarray:
    active_idx = np.flatnonzero(active)
    p = np.zeros_like(exposures, dtype=float)
    if len(active_idx) == 0:
        return p

    recurrent_mass = (exposures[active_idx] + EPS) / np.sum(exposures[active_idx] + EPS)
    uniform_mass = np.ones(len(active_idx)) / len(active_idx)
    p[active_idx] = rho * recurrent_mass + (1.0 - rho) * uniform_mass
    return p


def confidence(gamma: np.ndarray, h: np.ndarray) -> np.ndarray:
    return 1.0 - np.exp(-gamma * h)


def marginal_richness_gain(
    activities: ActivitySet,
    rho: float,
    resource_state: str,
) -> np.ndarray:
    p = future_relevance(activities.h, activities.active, rho)
    base = p * activities.q * activities.gamma * np.exp(-activities.gamma * activities.h)
    weights = RESOURCE_STATES[resource_state]
    penalty = weights["cost_weight"] * activities.cost + weights["risk_weight"] * activities.risk
    score = base - penalty
    score[~activities.active] = -np.inf
    return score


def make_activity_set(rng: np.random.Generator, K: int, active_initial: int | None = None) -> ActivitySet:
    gamma = rng.choice(
        [GAMMA_LEVELS["low"], GAMMA_LEVELS["medium"], GAMMA_LEVELS["high"]],
        size=K,
        p=[0.30, 0.40, 0.30],
    )
    q = np.clip(rng.normal(1.0, 0.18, K), 0.55, 1.45)
    cost = np.clip(rng.beta(2.0, 5.0, K), 0.03, 0.95)
    risk = np.clip(1.0 - gamma / max(GAMMA_LEVELS.values()) + rng.normal(0.0, 0.08, K), 0.03, 0.95)
    h = np.zeros(K, dtype=float)
    active = np.zeros(K, dtype=bool)
    if active_initial is None:
        active[:] = True
    else:
        active[:active_initial] = True
    return ActivitySet(gamma=gamma, q=q, cost=cost, risk=risk, h=h, active=active)


def maybe_arrive(rng: np.random.Generator, activities: ActivitySet, alpha: float) -> None:
    inactive = np.flatnonzero(~activities.active)
    if len(inactive) and rng.random() < alpha:
        activities.active[int(rng.choice(inactive))] = True


def choose_activity(
    rng: np.random.Generator,
    activities: ActivitySet,
    rho_policy: float,
    resource_state: str,
    tau: float | None = None,
) -> tuple[int, np.ndarray]:
    score = marginal_richness_gain(activities, rho_policy, resource_state)
    active_idx = np.flatnonzero(activities.active)

    if tau is None:
        max_score = np.max(score[active_idx])
        candidates = active_idx[score[active_idx] == max_score]
        return int(rng.choice(candidates)), score

    z = score[active_idx] / tau
    z -= np.max(z)
    probs = np.exp(z) / np.sum(np.exp(z))
    return int(rng.choice(active_idx, p=probs)), score


def richness_event(activities: ActivitySet, k: int) -> float:
    """Breadth/depth readout for the activity selected this week."""
    novelty = float(np.exp(-activities.gamma[k] * activities.h[k]))
    depth = float(activities.q[k] * confidence(np.array([activities.gamma[k]]), np.array([activities.h[k]]))[0])
    return 0.65 * novelty + 0.35 * depth


def depth_engagement(prior_h: float) -> float:
    """Depth readout: strongest for moderately familiar activities.

    This is intentionally not the same as new-arm discovery. It captures the
    L1 prediction that recurrent ecologies make moderately familiar activities
    adaptive: not completely novel, not fully routinized.
    """
    if 1 <= prior_h <= 4:
        return 1.0
    if prior_h > 4:
        return 0.25
    return 0.0


def curiosity_curve_rows() -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    K = 20
    h_values = np.arange(0, 16)
    for resource_state in RESOURCE_STATES:
        for gamma_label, gamma in GAMMA_LEVELS.items():
            for rho in RHO_VALUES:
                for h in h_values:
                    exposures = np.full(K, 2.0)
                    exposures[0] = float(h)
                    active = np.ones(K, dtype=bool)
                    p = future_relevance(exposures, active, rho)[0]
                    q = 1.0
                    cost = 0.25
                    risk = max(0.03, 1.0 - gamma / max(GAMMA_LEVELS.values()))
                    weights = RESOURCE_STATES[resource_state]
                    net = p * q * gamma * np.exp(-gamma * h)
                    net -= weights["cost_weight"] * cost + weights["risk_weight"] * risk
                    rows.append(
                        {
                            "simulation": "optimal_curiosity_curve",
                            "resource_state": resource_state,
                            "gamma_label": gamma_label,
                            "gamma": gamma,
                            "rho": rho,
                            "h": float(h),
                            "p_future": p,
                            "net_mrg": net,
                        }
                    )
    return rows


def simulate_policy(
    rng: np.random.Generator,
    ecology: str,
    resource_state: str,
    rho_policy: float,
    K_max: int = 40,
    T: int = T_WEEKS,
) -> dict[str, float | str]:
    if ecology == "recurrent":
        alpha = 0.04
        active_initial = 18
    elif ecology == "flowing":
        alpha = 0.35
        active_initial = 8
    else:
        raise ValueError(ecology)

    activities = make_activity_set(rng, K=K_max, active_initial=active_initial)
    chosen: list[int] = []
    chosen_prior_h: list[float] = []
    novelty_events: list[float] = []
    richness_events: list[float] = []
    scores: list[float] = []
    depth_events: list[float] = []

    for _week in range(T):
        maybe_arrive(rng, activities, alpha)
        k, score = choose_activity(rng, activities, rho_policy, resource_state)
        chosen.append(k)
        chosen_prior_h.append(float(activities.h[k]))
        novelty_events.append(float(activities.h[k] == 0))
        depth_events.append(depth_engagement(float(activities.h[k])))
        richness_events.append(richness_event(activities, k))
        scores.append(float(score[k]))
        activities.h[k] += 1.0

    counts = np.bincount(chosen, minlength=K_max)
    unique = int(np.count_nonzero(counts))
    novel = [h == 0 for h in chosen_prior_h]
    moderate = [1 <= h <= 4 for h in chosen_prior_h]
    routine = [h >= 5 for h in chosen_prior_h]
    active_count = activities.n_active

    return {
        "ecology": ecology,
        "resource_state": resource_state,
        "rho_policy": rho_policy,
        "arrival_rate": alpha,
        "active_final": float(active_count),
        "stock_richness": float(unique),
        "stock_share_active": float(unique / active_count),
        "flow_richness": float(mean(richness_events[-13:])),
        "early_flow_richness": float(mean(richness_events[:13])),
        "late_novelty_rate": float(mean(novelty_events[-13:])),
        "overall_novelty_rate": float(mean(novelty_events)),
        "depth_richness": float(mean(depth_events)),
        "top_arm_concentration": float(np.max(counts) / T),
        "mean_exposure_before_choice": float(mean(chosen_prior_h)),
        "proportion_novel_choices": float(mean(novel)),
        "proportion_moderate_choices": float(mean(moderate)),
        "proportion_routine_choices": float(mean(routine)),
        "mean_net_mrg": float(mean(scores)),
    }


def run_policy_fit(rng: np.random.Generator, n_sims: int) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for ecology, rho_policy in (("recurrent", 0.85), ("flowing", 0.05)):
        for resource_state in ("enriched", "depleted"):
            for sim in range(n_sims):
                row = simulate_policy(rng, ecology, resource_state, rho_policy)
                row["simulation"] = sim
                row["condition"] = f"{ecology}_{resource_state}"
                row["policy_type"] = "matched"
                rows.append(row)
    return rows


def run_policy_mismatch(rng: np.random.Generator, n_sims: int) -> list[dict[str, float | str]]:
    policies = {
        "recurrent_trained": 0.85,
        "flowing_trained": 0.05,
    }
    rows: list[dict[str, float | str]] = []
    for ecology in ("recurrent", "flowing"):
        for policy_name, rho_policy in policies.items():
            for sim in range(n_sims):
                row = simulate_policy(rng, ecology, "enriched", rho_policy)
                row["simulation"] = sim
                row["actual_ecology"] = ecology
                row["policy_name"] = policy_name
                rows.append(row)
    return rows


def summarize(rows: list[dict[str, float | str]], keys: tuple[str, ...]) -> list[dict[str, float | str]]:
    groups: dict[tuple[str, ...], list[dict[str, float | str]]] = {}
    for row in rows:
        key = tuple(str(row[k]) for k in keys)
        groups.setdefault(key, []).append(row)

    excluded = {"simulation", *keys}
    metrics = [k for k in rows[0] if k not in excluded and isinstance(rows[0][k], (int, float))]
    out: list[dict[str, float | str]] = []
    for key, group in sorted(groups.items()):
        item: dict[str, float | str] = dict(zip(keys, key))
        item["n_sims"] = len(group)
        for metric in metrics:
            values = [float(row[metric]) for row in group]
            item[f"{metric}_mean"] = mean(values)
            item[f"{metric}_sd"] = pstdev(values)
        out.append(item)
    return out


def write_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_curiosity_curves(rows: list[dict[str, float | str]], outdir: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.2), constrained_layout=True, sharex=True)
    for row_i, resource_state in enumerate(("enriched", "depleted")):
        for col_i, gamma_label in enumerate(("low", "medium", "high")):
            ax = axes[row_i, col_i]
            for rho in RHO_VALUES:
                subset = [
                    r
                    for r in rows
                    if r["resource_state"] == resource_state
                    and r["gamma_label"] == gamma_label
                    and float(r["rho"]) == rho
                ]
                subset = sorted(subset, key=lambda r: float(r["h"]))
                ax.plot(
                    [float(r["h"]) for r in subset],
                    [float(r["net_mrg"]) for r in subset],
                    label=f"rho={rho:g}",
                )
            ax.axhline(0, color="black", linewidth=0.8, alpha=0.35)
            ax.set_title(f"{resource_state}, gamma={gamma_label}")
            ax.grid(True, alpha=0.25)
            if row_i == 1:
                ax.set_xlabel("Exposure h")
            if col_i == 0:
                ax.set_ylabel("Net marginal richness gain")
    axes[0, 2].legend(frameon=False, fontsize=8)
    fig.savefig(outdir / "campus_l1_curiosity_curves.png", dpi=180)
    plt.close(fig)


def plot_policy_fit(summary_rows: list[dict[str, float | str]], outdir: Path) -> None:
    order = ["recurrent_enriched", "flowing_enriched", "recurrent_depleted", "flowing_depleted"]
    labels = ["recurrent\nenriched", "flowing\nenriched", "recurrent\ndepleted", "flowing\ndepleted"]
    metrics = [
        ("stock_richness_mean", "Stock richness"),
        ("flow_richness_mean", "Late flow richness"),
        ("depth_richness_mean", "Depth visit share"),
        ("proportion_moderate_choices_mean", "Moderate familiarity choices"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.4), constrained_layout=True)
    for ax, (metric, title) in zip(axes.flat, metrics):
        values = [
            float(next(r for r in summary_rows if r["condition"] == condition)[metric])
            for condition in order
        ]
        ax.bar(labels, values, color=["#4C78A8", "#54A24B", "#F58518", "#E45756"])
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.25)
    fig.savefig(outdir / "campus_l1_policy_fit.png", dpi=180)
    plt.close(fig)


def plot_policy_mismatch(summary_rows: list[dict[str, float | str]], outdir: Path) -> None:
    labels = []
    flow = []
    depth = []
    novelty = []
    for ecology in ("recurrent", "flowing"):
        for policy_name in ("recurrent_trained", "flowing_trained"):
            row = next(
                r
                for r in summary_rows
                if r["actual_ecology"] == ecology and r["policy_name"] == policy_name
            )
            labels.append(f"{policy_name.replace('_', ' ')}\nin {ecology}")
            flow.append(float(row["flow_richness_mean"]))
            depth.append(float(row["depth_richness_mean"]))
            novelty.append(float(row["late_novelty_rate_mean"]))

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10.5, 4.8), constrained_layout=True)
    width = 0.26
    ax.bar(x - width, flow, width=width, label="flow richness", color="#4C78A8")
    ax.bar(x, depth, width=width, label="depth richness", color="#F58518")
    ax.bar(x + width, novelty, width=width, label="late novelty", color="#54A24B")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Policy-environment mismatch")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(outdir / "campus_l1_policy_mismatch.png", dpi=180)
    plt.close(fig)


def make_report(
    fit_summary: list[dict[str, float | str]],
    mismatch_summary: list[dict[str, float | str]],
    outdir: Path,
    n_sims: int,
) -> None:
    def fit(condition: str) -> dict[str, float | str]:
        return next(r for r in fit_summary if r["condition"] == condition)

    def mismatch(ecology: str, policy: str) -> dict[str, float | str]:
        return next(
            r
            for r in mismatch_summary
            if r["actual_ecology"] == ecology and r["policy_name"] == policy
        )

    recurrent_enriched = fit("recurrent_enriched")
    flowing_enriched = fit("flowing_enriched")
    recurrent_depleted = fit("recurrent_depleted")
    flowing_depleted = fit("flowing_depleted")
    recurrent_matched = mismatch("recurrent", "recurrent_trained")
    recurrent_mismatch = mismatch("recurrent", "flowing_trained")
    flowing_matched = mismatch("flowing", "flowing_trained")
    flowing_mismatch = mismatch("flowing", "recurrent_trained")

    lines = [
        "# Campus L1 Model: New City / New Campus Life Exploration",
        "",
        "## Scenario",
        "",
        "Imagine you just moved to a new city or started at a new campus. Each week you choose one activity: cafes, museums, walking routes, clubs, research talks, volunteering, sports, language exchange, community events, creative workshops, or familiar routines.",
        "",
        "Each activity is a possible experience with exposure, future relevance, depth/learnability, richness potential, cost, risk/ambiguity, and subjective reward.",
        "",
        "## Computational-level objective",
        "",
        "This is not an online learning agent. The model computes which activity is rational to explore given the environment. The key score is net marginal richness gain:",
        "",
        "```text",
        "Net_MRG_k = p_k q_k gamma_k exp(-gamma_k h_k)",
        "            - lambda_s cost_k",
        "            - risk_weight_s risk_k",
        "```",
        "",
        "Future relevance is environment-shaped:",
        "",
        "```text",
        "p_k(t) = rho * (h_k + eps) / sum_j(h_j + eps)",
        "         + (1 - rho) * 1 / K_t",
        "```",
        "",
        "- high rho: recurrent ecology, past exposure predicts future relevance.",
        "- low rho: flowing/open ecology, future opportunities are less tied to past exposure.",
        "- enriched state: low cost and risk penalties.",
        "- depleted state: high cost and risk penalties.",
        "",
        "## Simulation 1: Optimal curiosity curve",
        "",
        "This simulation plots Net_MRG as a function of exposure h across rho, gamma, and resource state.",
        "",
        "Expected qualitative pattern:",
        "",
        "- high rho: value shifts toward moderately familiar activities.",
        "- low rho: value is highest for novel/low-exposure activities.",
        "- depleted state: the whole curve shifts downward because cost and ambiguity matter more.",
        "",
        "Figure: `results/campus_l1_curiosity_curves.png`",
        "",
        "## Simulation 2: Policy-environment fit",
        "",
        f"Each condition uses {n_sims} simulations over {T_WEEKS} weeks.",
        "",
        "| Ecology / resource | Stock richness | Late flow richness | Depth richness | Novel choices | Moderate choices | Routine choices |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, row in (
        ("recurrent / enriched", recurrent_enriched),
        ("flowing / enriched", flowing_enriched),
        ("recurrent / depleted", recurrent_depleted),
        ("flowing / depleted", flowing_depleted),
    ):
        lines.append(
            "| {label} | {stock:.2f} | {flow:.3f} | {depth:.3f} | {novel:.3f} | {moderate:.3f} | {routine:.3f} |".format(
                label=label,
                stock=float(row["stock_richness_mean"]),
                flow=float(row["flow_richness_mean"]),
                depth=float(row["depth_richness_mean"]),
                novel=float(row["proportion_novel_choices_mean"]),
                moderate=float(row["proportion_moderate_choices_mean"]),
                routine=float(row["proportion_routine_choices_mean"]),
            )
        )

    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "- Recurrent/enriched ecologies support depth and moderate-familiarity exploration.",
            "- Flowing/enriched ecologies support novelty seeking and higher late flow.",
            "- Depleted states reduce exploration value even when opportunities exist.",
            "",
            "Figure: `results/campus_l1_policy_fit.png`",
            "",
            "## Simulation 3: Policy-environment mismatch",
            "",
            "Two policies were optimized for different ecology assumptions:",
            "",
            "- recurrent-trained policy: assumes past exposure predicts future relevance.",
            "- flowing-trained policy: assumes future opportunities are less tied to past exposure.",
            "",
            "| Actual ecology | Policy | Late flow richness | Depth richness | Late novelty rate | Top-arm concentration |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for ecology, policy, row in (
        ("recurrent", "recurrent-trained", recurrent_matched),
        ("recurrent", "flowing-trained", recurrent_mismatch),
        ("flowing", "flowing-trained", flowing_matched),
        ("flowing", "recurrent-trained", flowing_mismatch),
    ):
        lines.append(
            "| {ecology} | {policy} | {flow:.3f} | {depth:.3f} | {novelty:.3f} | {top:.3f} |".format(
                ecology=ecology,
                policy=policy,
                flow=float(row["flow_richness_mean"]),
                depth=float(row["depth_richness_mean"]),
                novelty=float(row["late_novelty_rate_mean"]),
                top=float(row["top_arm_concentration_mean"]),
            )
        )

    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "- A recurrent-trained policy in a flowing ecology under-explores new opportunities relative to a flowing-trained policy.",
            "- A flowing-trained policy in a recurrent ecology chases novelty and gives up some depth.",
            "- Psychological richness is therefore partly policy-environment fit, not only environmental abundance.",
            "",
            "Figure: `results/campus_l1_policy_mismatch.png`",
            "",
            "## Main conclusion",
            "",
            "The campus/city L1 model reframes richness as adaptive exploration under ecological structure and resource constraints. Richness is highest when the policy fits the ecology: recurrent ecologies reward depth and moderate familiarity; flowing ecologies reward novelty seeking; depleted states gate exploration even when opportunities are objectively present.",
        ]
    )
    (outdir / "campus_l1_report.md").write_text("\n".join(lines) + "\n")


def run(outdir: Path, n_sims: int) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    curve_rows = curiosity_curve_rows()
    fit_rows = run_policy_fit(rng, n_sims)
    mismatch_rows = run_policy_mismatch(rng, n_sims)
    fit_summary = summarize(fit_rows, ("condition", "ecology", "resource_state"))
    mismatch_summary = summarize(mismatch_rows, ("actual_ecology", "policy_name"))

    write_csv(outdir / "campus_l1_curiosity_curves.csv", curve_rows)
    write_csv(outdir / "campus_l1_policy_fit.csv", fit_rows)
    write_csv(outdir / "campus_l1_policy_fit_summary.csv", fit_summary)
    write_csv(outdir / "campus_l1_policy_mismatch.csv", mismatch_rows)
    write_csv(outdir / "campus_l1_policy_mismatch_summary.csv", mismatch_summary)

    plot_curiosity_curves(curve_rows, outdir)
    plot_policy_fit(fit_summary, outdir)
    plot_policy_mismatch(mismatch_summary, outdir)
    make_report(fit_summary, mismatch_summary, outdir, n_sims)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=Path("results"))
    parser.add_argument("--sims", type=int, default=500)
    args = parser.parse_args()
    run(args.outdir, args.sims)
    print(f"Wrote campus L1 outputs to {args.outdir.resolve()}")


if __name__ == "__main__":
    main()
