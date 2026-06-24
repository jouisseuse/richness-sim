# From Ecological Opportunity to Experienced Richness

**Subtitle:** Why more possible lives do not always produce richer lived lives

## Overview

This project uses simple bandit simulations to examine whether environments with more possible experiences automatically produce richer lived experiences.

The central result is that ecological richness does **not** automatically become experienced richness. A better framework is:

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

## Model

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

The main outcomes are richness measures rather than reward:

- final experienced richness
- late novelty rate
- novelty decay
- stock richness
- flow richness

## Simulation 1: Static vs Flowing Ecology

This simulation compares two types of rich ecology:

- **Static ecology**: all options exist from the beginning.
- **Flowing ecology**: new options continue to appear over time.

| Ecology type | Final experienced richness | Late novelty rate | Novelty decay |
|---|---:|---:|---:|
| static | 71.4 | 0.155 | 0.537 |
| flowing | 64.0 | 0.217 | 0.183 |

Interpretation:

Static ecology produces higher final stock richness, but flowing ecology better preserves late novelty. Rich environments matter most when they continuously generate new feasible opportunities, not simply when they contain many options at the start.

## Simulation 2: Exploration Policy Robustness

Three representative learning policies were compared in a rich flowing ecology:

| Policy | Algorithm | Final experienced richness | Late novelty rate | Top-arm concentration |
|---|---|---:|---:|---:|
| low exploration | epsilon-greedy 0.1 | 21.2 | 0.101 | 0.896 |
| adaptive exploration | Thompson sampling | 63.2 | 0.213 | 0.091 |
| forced exploration | epsilon-greedy 0.5 | 97.3 | 0.440 | 0.515 |

Interpretation:

Different learning policies produce different kinds of rich lives. Low exploration routinizes quickly. Forced exploration maximizes novelty, but it is an upper-bound case. Thompson sampling is the most natural middle case: Bayesian, reward-sensitive, and still capable of producing ongoing novelty.

## Simulation 3: Breadth vs Depth

The first two simulations define richness mainly as breadth: trying new arms. This extension adds depth.

Two arm types are introduced:

- **Surface arms**: highly novel the first time, but novelty decays quickly.
- **Deep arms**: repeated engagement gradually increases meaning or richness.

| Environment | K | Unique arms | Unique/K | Late new-discovery rate | Late richness flow | Cumulative richness |
|---|---:|---:|---:|---:|---:|---:|
| breadth city | 100 | 71.7 | 0.717 | 0.149 | 0.235 | 74.1 |
| depth town | 10 | 10.0 | 0.999 | 0.001 | 0.784 | 134.1 |

Interpretation:

If richness includes depth, small ecologies are not automatically poor. Repeated engagement with deep experiences can sustain meaning even when new-arm discovery is low. This prevents the model from being too biased toward urban consumption and option breadth.

## Main Conclusions

1. **Ecological richness does not automatically become experienced richness.** Having many possible opportunities does not mean agents will live many different experiences.

2. **The conversion pathway matters.** Time, cost, and access can block the pathway from ecological richness to feasible richness to experienced richness.

3. **Stock richness and flow richness are different.** Stock richness is accumulated diversity; flow richness is ongoing novelty. Psychological descriptors such as interesting, eventful, dramatic, and perspective-changing may be closer to flow richness.

4. **Rich ecologies matter most when they generate continuing feasible novelty.** The psychologically rich environment is not simply the place with the most options. It is the place where new options continue to become livable, affordable, and explorable.

5. **Richness can come from breadth or depth.** Breadth means sampling many different experiences. Depth means repeated engagement with experiences that continue to become meaningful.

## Suggested Framing

Recommended title:

**From Ecological Opportunity to Experienced Richness**

Subtitle:

**Why More Possible Lives Do Not Always Produce Richer Lived Lives**

Alternative theory-forward title:

**Stock and Flow in Psychological Richness**

## Files

- `focused_simulations.py`: final focused simulations.
- `simulate_richness.py`: shared utilities and broader exploratory model.
- `results/focused_simulations.png`: compact summary figure.
- `results/focused_stock_flow_summary.csv`: static vs flowing ecology summary.
- `results/focused_policy_summary.csv`: exploration policy summary.
- `results/focused_breadth_depth_summary.csv`: breadth vs depth summary.
