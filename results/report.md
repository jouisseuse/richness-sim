# Ecological, Feasible, and Experienced Richness Report

## Model

- Ground truth reward: every arm is Bernoulli(p=0.9).
- Prior belief for every arm: Beta(1, 1).
- Bayesian update: success increments alpha; failure increments beta.
- Algorithms: epsilon-greedy with epsilon = 0.1, 0.5, 0.9; Thompson sampling; UCB.
- Single-agent baseline; rounds = 200; simulations per condition = 100.
- Each arm has time cost, money cost, and novelty. Novelty is initially 1 and becomes 0 after the arm is tried.
- Agents can only choose arms within both their time budget and money budget.

## Conditions

| Condition | Ecology | Time/cost | Example | K | Time budget | Money budget | Arm costs |
|---|---|---|---|---:|---:|---:|---|
| big_city | rich | low_time_high_cost | big city | 100 | 0.35 | 0.35 | high |
| small_town | poor | high_time_low_cost | small town | 10 | 0.80 | 0.80 | low |
| ideal_city | rich | high_time_low_cost | ideal city / campus | 100 | 0.80 | 0.80 | low |
| constrained_rural | poor | low_time_high_cost | constrained rural / isolated work life | 10 | 0.35 | 0.35 | high |

## Main Results: thompson

| Condition | Ecological | Feasible | Experienced | Unaffordable | Unexplored | Feasible/Ecological | Experienced/Feasible | Experienced/Ecological | Entropy | Avg reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| big_city | 100 | 19.0 | 18.8 | 81.0 | 81.2 | 0.190 | 0.986 | 0.188 | 2.53 | 0.896 |
| small_town | 10 | 10.0 | 10.0 | 0.0 | 0.0 | 1.000 | 0.998 | 0.998 | 1.92 | 0.902 |
| ideal_city | 100 | 99.8 | 71.4 | 0.2 | 28.6 | 0.998 | 0.715 | 0.714 | 4.01 | 0.902 |
| constrained_rural | 10 | 2.0 | 2.0 | 8.0 | 8.0 | 0.200 | 1.000 | 0.200 | 0.37 | 0.901 |

## Interpretation

- Big city has the highest ecological richness among realistic conditions (K=100), but only 19.0 options were feasible on average under low time and high cost.
- Small town has lower ecological richness (K=10), but its feasible/ecological conversion was 100.0%, versus 19.0% in big city.
- Ideal city/campus is the richest lived condition: many options plus enough time and affordable cost produced 71.4 experienced arms on average.
- Big city still produced more absolute experienced richness than small town in this calibration (18.8 vs 10.0), but it also produced many more unrealized possibilities: 81.0 unaffordable and 81.2 unexplored options.
- The central pattern is not simply rich ecology -> rich experience. The conversion pathway matters: ecological richness -> feasible richness -> experienced richness.

One-sentence takeaway: rich ecology may increase imagined possibilities more than lived experiences.

## Temporal Richness Analysis

This sub-experiment holds time and money budgets generous and affordable, then varies ecological K = 10, 25, 50, 100 under Thompson sampling. It asks whether richer ecology changes the timing of novelty, not just the final Round 200 total.

| K | Unique at round 20 | Unique at round 100 | Unique at round 200 | Unique/K at round 200 | Discovery rate rounds 1-50 | Discovery rate rounds 151-200 |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 7.4 | 9.8 | 10.0 | 1.000 | 0.183 | 0.001 |
| 25 | 11.8 | 21.8 | 24.3 | 0.971 | 0.361 | 0.015 |
| 50 | 14.7 | 35.5 | 43.4 | 0.868 | 0.528 | 0.061 |
| 100 | 17.1 | 53.0 | 71.2 | 0.712 | 0.697 | 0.145 |

- Figure 1 tests cumulative experienced richness. Larger ecologies maintain a higher absolute ceiling, so K=100 keeps accumulating more unique experiences than K=10.
- Figure 2 is the closest proxy for ongoing psychological richness: in K=100, the new-discovery rate falls from 0.697 in rounds 1-50 to 0.145 in rounds 151-200, showing routine formation even in rich ecology.
- Figure 3 shows the absolute/relative split: by round 200, K=10 reaches 100.0% of its ecology, while K=100 reaches 71.2%. Rich ecology creates more total novelty, but a smaller fraction of possible life is actually lived.

## Algorithm And Flowing-Arm Checks

| Algorithm | K=100 fixed: unique at round 200 | K=100 fixed: late discovery rate |
|---|---:|---:|
| egreedy_0.1 | 21.0 | 0.100 |
| egreedy_0.5 | 97.4 | 0.435 |
| thompson | 71.2 | 0.145 |
| ucb | 99.9 | 0.000 |

- In K=100 under Thompson sampling, flowing arms keep late novelty higher than fixed arms (0.209 vs 0.145). This captures environments where new possibilities keep appearing rather than all existing from the start.
- Figure 4 compares algorithms. Figure 5 compares fixed versus flowing arms.

## Sensitivity Sweep

The broader sweep crosses K = 10, 25, 50, 100; budget = tight, moderate, generous; cost profile = low/high; algorithm = e-greedy 0.1, e-greedy 0.5, Thompson, UCB; and arm regime = fixed/flowing. Each cell uses 30 simulations.

Top conditions for sustained late novelty:

| Rank | K | Budget | Cost | Algorithm | Arms | Experienced/Ecological | Late discovery | Novelty decay |
|---:|---:|---|---|---|---|---:|---:|---:|
| 1 | 100 | generous | low | egreedy_0.5 | flowing | 0.975 | 0.431 | 0.091 |
| 2 | 100 | generous | low | egreedy_0.5 | fixed | 0.972 | 0.425 | 0.087 |
| 3 | 100 | moderate | low | egreedy_0.5 | flowing | 0.935 | 0.357 | 0.154 |
| 4 | 100 | moderate | low | egreedy_0.5 | fixed | 0.935 | 0.327 | 0.193 |
| 5 | 100 | generous | high | egreedy_0.5 | fixed | 0.912 | 0.297 | 0.213 |
| 6 | 100 | generous | high | egreedy_0.5 | flowing | 0.906 | 0.286 | 0.241 |
| 7 | 100 | generous | low | thompson | flowing | 0.628 | 0.209 | 0.187 |
| 8 | 100 | moderate | low | thompson | flowing | 0.614 | 0.207 | 0.185 |

Top conditions for total ecological conversion:

| Rank | K | Budget | Cost | Algorithm | Arms | Experienced/Ecological | Late discovery | Feasible/Ecological |
|---:|---:|---|---|---|---|---:|---:|---:|
| 1 | 10 | generous | low | egreedy_0.5 | fixed | 1.000 | 0.000 | 1.000 |
| 2 | 10 | generous | low | egreedy_0.5 | flowing | 1.000 | 0.000 | 1.000 |
| 3 | 25 | generous | low | egreedy_0.5 | fixed | 1.000 | 0.000 | 1.000 |
| 4 | 25 | generous | low | egreedy_0.5 | flowing | 1.000 | 0.000 | 1.000 |
| 5 | 50 | generous | low | egreedy_0.5 | fixed | 0.999 | 0.000 | 0.999 |
| 6 | 100 | generous | low | ucb | fixed | 0.999 | 0.000 | 0.999 |
| 7 | 100 | generous | low | ucb | flowing | 0.999 | 0.000 | 0.999 |
| 8 | 25 | generous | low | ucb | fixed | 0.999 | 0.000 | 0.999 |

## Best Interpretation

- The most theoretically useful distinction is not high versus low final richness, but **stock novelty versus flow novelty**.
- UCB maximizes stock conversion in rich, affordable environments: for K=100/generous/low-cost/flowing arms, it reaches 99.9% experienced/ecological conversion, but its late discovery rate is 0.000. In other words, it exhausts the space and then novelty stops.
- E-greedy 0.5 keeps the strongest late novelty in the same environment (0.431), but partly because it imposes constant random exploration. This is useful as an upper-bound exploration regime, not necessarily the most psychologically realistic learner.
- Thompson sampling is the best middle model for the theory: it is Bayesian, reward-sensitive, and still produces ongoing novelty when arms flow into the ecology. In K=100/generous/low-cost/flowing arms, it reaches 62.8% total conversion and maintains late novelty at 0.209.
- The interpretation I would foreground is: rich ecologies matter most when they maintain a **continuing arrival process** of feasible options. Static abundance quickly becomes a stock of known possibilities; flowing abundance better captures eventfulness.


## Output Files

- `results/raw_simulations.csv`: one row per simulation run.
- `results/summary_results.csv`: aggregated means and standard deviations.
- `results/richness_layers_thompson.png`: ecological, feasible, and experienced richness visualization.
- `results/temporal_raw.csv`: one row per temporal simulation round.
- `results/temporal_summary.csv`: average temporal trajectories by K.
- `results/figure1_cumulative_unique.png`: cumulative unique arms over time.
- `results/figure2_new_discoveries.png`: new arms discovered per round over time.
- `results/figure3_unique_share.png`: cumulative unique arms divided by K over time.
- `results/figure4_algorithm_comparison.png`: algorithm comparison for K=100.
- `results/figure5_flowing_arms.png`: fixed versus flowing arms comparison.
- `results/figure6_sensitivity_late_novelty.png`: late novelty across budget/cost/algorithm settings.
- `results/figure7_conversion_vs_late_novelty.png`: tradeoff between total conversion and ongoing novelty.
- `results/sensitivity_raw.csv`: broad parameter sweep raw results.
- `results/sensitivity_summary.csv`: broad parameter sweep summary.
