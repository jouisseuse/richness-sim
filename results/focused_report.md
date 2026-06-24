# Focused Simulation Report: From Ecological Opportunity to Experienced Richness

## Scope

This focused report uses 300 simulations for stock/flow and policy tests, plus 700 simulations for the breadth/depth extension. It deliberately avoids broad parameter search and keeps only three targeted simulations.

## Simulation 1: Static vs Flowing Ecology

| Ecology type | Final experienced richness | Late novelty rate | Novelty decay |
|---|---:|---:|---:|
| static | 71.4 | 0.155 | 0.537 |
| flowing | 64.0 | 0.217 | 0.183 |

Interpretation: static abundance mainly creates stock richness. Flowing ecology is more important for flow richness because feasible new options keep arriving.

## Simulation 2: Exploration Policy Robustness

| Policy | Algorithm | Final experienced richness | Late novelty rate | Top-arm concentration |
|---|---|---:|---:|---:|
| low exploration | e-greedy 0.1 | 21.2 | 0.101 | 0.896 |
| adaptive exploration | Thompson | 63.2 | 0.213 | 0.091 |
| forced exploration | e-greedy 0.5 | 97.3 | 0.440 | 0.515 |

Interpretation: different learning policies produce different kinds of rich lives. Low exploration routinizes, forced exploration maximizes novelty, and Thompson is the most psychologically plausible middle case.

## Simulation 3: Breadth vs Depth

| Environment | K | Unique arms | Unique/K | Late new-discovery rate | Late richness flow | Cumulative richness |
|---|---:|---:|---:|---:|---:|---:|
| breadth city | 100 | 71.7 | 0.717 | 0.149 | 0.235 | 74.1 |
| depth town | 10 | 10.0 | 0.999 | 0.001 | 0.784 | 134.1 |

Interpretation: if richness includes depth, small ecologies are not automatically poor. Repeated engagement with deep arms can sustain meaning even when new-arm discovery is low.

## Four Conclusions

1. Ecological richness does not automatically become experienced richness. Ecological opportunity is not the same as lived experience.
2. The conversion pathway matters: ecological richness -> feasible richness -> experienced richness. Time, cost, and access can block conversion.
3. Stock richness and flow richness are different. Stock richness is accumulated diversity; flow richness is ongoing novelty.
4. Rich ecologies matter most when they generate continuing feasible novelty. The psychologically rich environment is not simply the place with the most options, but the place where new options keep becoming livable.

## Suggested Title

**From Ecological Opportunity to Experienced Richness**

Subtitle: **Why more possible lives do not always produce richer lived lives**

Alternative theory-forward title: **Stock and Flow in Psychological Richness**

## Files

- `results/focused_stock_flow.csv`
- `results/focused_policy.csv`
- `results/focused_breadth_depth.csv`
- `results/focused_simulations.png`
