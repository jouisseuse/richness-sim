# From Ecological Opportunity to Experienced Richness

**Subtitle:** Why more possible lives do not always produce richer lived lives

## 1. Overview

This project uses simple bandit simulations to examine whether environments with more possible experiences automatically produce richer lived experiences.

The central result is that ecological richness does **not** automatically become experienced richness. A useful framework is:

```text
Ecological richness
-> Feasible richness
-> Experienced richness
-> Stock richness / Flow richness
```

Definitions:

- **Ecological richness**: all possible opportunities in an environment.
- **Feasible richness**: opportunities that are accessible given constraints such as time and cost.
- **Experienced richness**: opportunities that are actually chosen and explored.
- **Stock richness**: accumulated diversity of experiences.
- **Flow richness**: ongoing novelty over time.

The main theoretical claim is:

> Psychological richness may emerge from the conversion of ecological possibilities into lived novelty.

## 2. Model

Each arm represents a possible experience. All arms have the same objective reward:

```text
reward ~ Bernoulli(p = 0.9)
```

Agents update beliefs using a Beta-Bernoulli Bayesian rule:

```text
prior: Beta(1, 1)
success: alpha += 1
failure: beta += 1
```

The primary outcomes are not reward, because all arms are equally good in expectation. The primary outcomes are richness measures:

- unique arms tried
- feasible richness
- experienced richness
- conversion rates
- late novelty rate
- novelty decay
- stock richness
- flow richness

## 3. Ecological, Feasible, and Experienced Richness

The first simulation compares four stylized environments:

| Condition | Ecology | Constraint | Example | K | Time budget | Money budget | Arm costs |
|---|---|---|---|---:|---:|---:|---|
| big city | rich | low time, high cost | large city | 100 | 0.35 | 0.35 | high |
| small town | poor | high time, low cost | small town | 10 | 0.80 | 0.80 | low |
| ideal city | rich | high time, low cost | campus / walkable city | 100 | 0.80 | 0.80 | low |
| constrained rural | poor | low time, high cost | isolated work life | 10 | 0.35 | 0.35 | high |

Main Thompson-sampling results:

| Condition | Ecological | Feasible | Experienced | Feasible/Ecological | Experienced/Ecological |
|---|---:|---:|---:|---:|---:|
| big city | 100 | 19.0 | 18.8 | 0.190 | 0.188 |
| small town | 10 | 10.0 | 10.0 | 1.000 | 0.998 |
| ideal city | 100 | 99.8 | 71.4 | 0.998 | 0.714 |
| constrained rural | 10 | 2.0 | 2.0 | 0.200 | 0.200 |

Interpretation:

A big city has many possible experiences, but many are not feasible under time and cost constraints. A small town has fewer possible experiences, but a much higher conversion rate from possible experiences to lived experiences. The richest lived condition is the ideal city/campus case: many options plus enough time and affordable costs.

Key figure:

- [richness_layers_thompson.png](richness_layers_thompson.png)

## 4. Temporal Richness

The second simulation asks whether richness should be measured only at the final round. It varies ecological size:

```text
K = 10, 25, 50, 100
```

Main results:

| K | Unique at round 20 | Unique at round 100 | Unique at round 200 | Unique/K at round 200 | Discovery rate rounds 1-50 | Discovery rate rounds 151-200 |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 7.4 | 9.8 | 10.0 | 1.000 | 0.183 | 0.001 |
| 25 | 11.8 | 21.8 | 24.3 | 0.971 | 0.361 | 0.015 |
| 50 | 14.7 | 35.5 | 43.4 | 0.868 | 0.528 | 0.061 |
| 100 | 17.1 | 53.0 | 71.2 | 0.712 | 0.697 | 0.145 |

Interpretation:

Larger ecologies create more accumulated experience, but agents experience a smaller fraction of the total space. Novelty also decays over time. Even in the K=100 ecology, new-discovery rate falls from 0.697 in rounds 1-50 to 0.145 in rounds 151-200.

This motivates the distinction between:

- **stock richness**: how much diversity has accumulated
- **flow richness**: whether new experiences are still happening

Key figures:

- [figure1_cumulative_unique.png](figure1_cumulative_unique.png)
- [figure2_new_discoveries.png](figure2_new_discoveries.png)
- [figure3_unique_share.png](figure3_unique_share.png)

## 5. Focused Simulation 1: Static vs Flowing Ecology

This is the most important focused simulation.

- **Static ecology**: all options exist from the beginning.
- **Flowing ecology**: new options continue to appear over time.

| Ecology type | Final experienced richness | Late novelty rate | Novelty decay |
|---|---:|---:|---:|
| static | 71.4 | 0.155 | 0.537 |
| flowing | 64.0 | 0.217 | 0.183 |

Interpretation:

Static ecology produces more final stock richness, but flowing ecology better preserves late novelty. This suggests that rich environments matter most when they continuously generate new feasible opportunities, not simply when they contain many options at the start.

Key figures:

- [focused_simulations.png](focused_simulations.png)
- [figure5_flowing_arms.png](figure5_flowing_arms.png)

## 6. Focused Simulation 2: Exploration Policy Robustness

Three representative learning policies were compared:

| Policy | Algorithm | Final experienced richness | Late novelty rate | Top-arm concentration |
|---|---|---:|---:|---:|
| low exploration | epsilon-greedy 0.1 | 21.2 | 0.101 | 0.896 |
| adaptive exploration | Thompson sampling | 63.2 | 0.213 | 0.091 |
| forced exploration | epsilon-greedy 0.5 | 97.3 | 0.440 | 0.515 |

Interpretation:

Different learning policies produce different types of rich lives. Low exploration routinizes quickly. Forced exploration maximizes novelty, but it is an upper-bound case. Thompson sampling is the most natural middle case: Bayesian, reward-sensitive, and still capable of producing ongoing novelty.

Key figure:

- [figure4_algorithm_comparison.png](figure4_algorithm_comparison.png)

## 7. Focused Simulation 3: Breadth vs Depth

The earlier simulations define richness mainly as breadth: trying new arms. This extension adds depth.

Two arm types are introduced:

- **Surface arms**: highly novel the first time, but novelty decays quickly.
- **Deep arms**: repeated engagement gradually increases meaning or richness.

| Environment | K | Unique arms | Unique/K | Late new-discovery rate | Late richness flow | Cumulative richness |
|---|---:|---:|---:|---:|---:|---:|
| breadth city | 100 | 71.7 | 0.717 | 0.149 | 0.235 | 74.1 |
| depth town | 10 | 10.0 | 0.999 | 0.001 | 0.784 | 134.1 |

Interpretation:

If richness includes depth, small ecologies are not automatically poor. Repeated engagement with deep experiences can sustain meaning even when new-arm discovery is low. This prevents the model from being too biased toward urban consumption and option breadth.

## 8. Robustness Sweep

A broader sensitivity sweep varied:

- K: 10, 25, 50, 100
- budget: tight, moderate, generous
- cost profile: low, high
- algorithm: epsilon-greedy 0.1, epsilon-greedy 0.5, Thompson sampling, UCB
- arm regime: fixed, flowing

Selected K=100, generous-budget, low-cost, flowing-ecology results:

| Algorithm | Experienced/Ecological | Late discovery rate |
|---|---:|---:|
| epsilon-greedy 0.5 | 0.975 | 0.431 |
| Thompson sampling | 0.628 | 0.209 |
| UCB | 0.999 | 0.000 |

Interpretation:

UCB exhausts the space and therefore maximizes stock conversion, but late novelty disappears. Epsilon-greedy 0.5 maintains high novelty through forced exploration. Thompson sampling is the best theory-facing middle case because it is adaptive and still maintains some flow novelty when new feasible options continue to appear.

Key figures:

- [figure6_sensitivity_late_novelty.png](figure6_sensitivity_late_novelty.png)
- [figure7_conversion_vs_late_novelty.png](figure7_conversion_vs_late_novelty.png)

## 9. Main Conclusions

### Conclusion 1: Ecological richness does not automatically become experienced richness.

Having many possible opportunities does not mean that agents will live many different experiences.

```text
Ecological opportunity != lived experience
```

### Conclusion 2: The conversion pathway matters.

The core pathway is:

```text
Ecological richness
-> Feasible richness
-> Experienced richness
```

Time, cost, and access can block conversion. A city can have high ecological richness but low feasible conversion. A smaller ecology can have lower ecological richness but higher conversion.

### Conclusion 3: Stock richness and flow richness are different.

```text
Stock richness = accumulated diversity
Flow richness = ongoing novelty
```

Psychological descriptors such as interesting, eventful, dramatic, and perspective-changing may be closer to flow richness than final unique-arm counts.

### Conclusion 4: Rich ecologies matter most when they generate continuing feasible novelty.

The psychologically rich environment is not simply the place with the most options. It is the place where new options continue to become livable, affordable, and explorable.

Examples include:

- campuses
- walkable mixed-use neighborhoods
- creative communities
- research environments
- dense but affordable social ecologies

### Conclusion 5: Richness can come from breadth or depth.

Breadth means sampling many different experiences. Depth means repeated engagement with experiences that continue to become meaningful. A small ecology can still support psychological richness if it contains deep experiences.

## 10. Suggested Framing

Recommended title:

**From Ecological Opportunity to Experienced Richness**

Subtitle:

**Why More Possible Lives Do Not Always Produce Richer Lived Lives**

Alternative theory-forward title:

**Stock and Flow in Psychological Richness**

## 11. Files for Reproduction

Scripts:

- [simulate_richness.py](../simulate_richness.py)
- [focused_simulations.py](../focused_simulations.py)

Key result files:

- [focused_report.md](focused_report.md)
- [report.md](report.md)
- [focused_stock_flow_summary.csv](focused_stock_flow_summary.csv)
- [focused_policy_summary.csv](focused_policy_summary.csv)
- [focused_breadth_depth_summary.csv](focused_breadth_depth_summary.csv)
- [summary_results.csv](summary_results.csv)
- [temporal_summary.csv](temporal_summary.csv)
- [sensitivity_summary.csv](sensitivity_summary.csv)

