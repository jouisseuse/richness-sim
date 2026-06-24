# From Ecological Opportunity to Experienced Richness

Subtitle: Why more possible lives do not always produce richer lived lives

## Executive Summary

本项目用 bandit simulation 检验一个核心问题：

**Does ecological richness automatically become psychological or experienced richness?**

结论是否定的。环境里有很多 options，并不意味着 agent 会真正经历很多东西。更好的框架是：

```text
Ecological richness
-> Feasible richness
-> Experienced richness
-> Stock richness / Flow richness
```

其中：

- **Ecological richness**: 环境中存在多少可能经验。
- **Feasible richness**: 在时间、金钱、成本约束下，agent 实际负担得起多少经验。
- **Experienced richness**: agent 实际选择并经历了多少经验。
- **Stock richness**: 到目前为止累计经历过多少不同经验。
- **Flow richness**: 最近是否仍在持续遇到新经验。

最重要的理论贡献是 stock richness 和 flow richness 的区分。心理上的 interesting, eventful, novel, perspective-changing 可能更接近 **flow richness**，而不只是最终累计 unique arms。

## Model

所有主要模拟都使用同一个基本学习模型：

- 每个 arm 代表一种 possible experience。
- 所有 arm 的真实 reward 相同：Bernoulli `p = 0.9`。
- agent 的 prior belief 是 `Beta(1, 1)`。
- Bayes update: success 增加 `alpha`，failure 增加 `beta`。
- agent 只能选择 time budget 和 money budget 内可负担的 arms。
- 主要 outcomes 不是 reward，而是 richness metrics。

主要算法：

- low exploration: `epsilon-greedy 0.1`
- adaptive exploration: Thompson sampling
- forced exploration: `epsilon-greedy 0.5`
- robustness 中也比较了 UCB

## Main Simulation 1: Ecological, Feasible, Experienced Richness

这组模拟比较四种环境：

| Condition | Ecology | Time/cost | Example | K | Time budget | Money budget | Arm costs |
|---|---|---|---|---:|---:|---:|---|
| big_city | rich | low time, high cost | big city | 100 | 0.35 | 0.35 | high |
| small_town | poor | high time, low cost | small town | 10 | 0.80 | 0.80 | low |
| ideal_city | rich | high time, low cost | campus / ideal city | 100 | 0.80 | 0.80 | low |
| constrained_rural | poor | low time, high cost | isolated rural work life | 10 | 0.35 | 0.35 | high |

Thompson sampling 结果：

| Condition | Ecological | Feasible | Experienced | Unaffordable | Unexplored | Feasible/Ecological | Experienced/Ecological |
|---|---:|---:|---:|---:|---:|---:|---:|
| big_city | 100 | 19.0 | 18.8 | 81.0 | 81.2 | 0.190 | 0.188 |
| small_town | 10 | 10.0 | 10.0 | 0.0 | 0.0 | 1.000 | 0.998 |
| ideal_city | 100 | 99.8 | 71.4 | 0.2 | 28.6 | 0.998 | 0.714 |
| constrained_rural | 10 | 2.0 | 2.0 | 8.0 | 8.0 | 0.200 | 0.200 |

Interpretation:

大城市拥有最高 ecological richness，但在低时间、高成本条件下，只有约 19 个 options 是 feasible。小地方 K 低，但 feasible conversion 很高。最 rich 的不是普通大城市，而是 **many options + enough time + affordable cost** 的环境，比如 campus、walkable mixed-use neighborhood、creative community、research environment。

Figure:

- [richness_layers_thompson.png](/home/ubuntu/richness/results/richness_layers_thompson.png)

## Main Simulation 2: Temporal Richness

这组模拟固定 generous budget 和 low cost，只改变 ecological K:

`K = 10, 25, 50, 100`

核心问题不是 round 200 的最终值，而是 novelty 如何随时间变化。

| K | Unique at round 20 | Unique at round 100 | Unique at round 200 | Unique/K at round 200 | Discovery rate 1-50 | Discovery rate 151-200 |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 7.4 | 9.8 | 10.0 | 1.000 | 0.183 | 0.001 |
| 25 | 11.8 | 21.8 | 24.3 | 0.971 | 0.361 | 0.015 |
| 50 | 14.7 | 35.5 | 43.4 | 0.868 | 0.528 | 0.061 |
| 100 | 17.1 | 53.0 | 71.2 | 0.712 | 0.697 | 0.145 |

Interpretation:

Rich ecology 提高 absolute experienced richness，但 relative richness 下降。K=100 到 round 200 体验了 71.2 个 arms，但这只是全部 possible arms 的 71.2%。K=10 几乎完全探索完。

更关键的是 discovery rate 衰减。K=100 的 new discovery rate 从前 50 轮的 `0.697` 降到后 50 轮的 `0.145`。这说明即使在 rich ecology 中，agent 也会逐渐形成 routines。

Figures:

- [figure1_cumulative_unique.png](/home/ubuntu/richness/results/figure1_cumulative_unique.png)
- [figure2_new_discoveries.png](/home/ubuntu/richness/results/figure2_new_discoveries.png)
- [figure3_unique_share.png](/home/ubuntu/richness/results/figure3_unique_share.png)

## Focused Simulation 1: Static vs Flowing Ecology

这是最重要的 focused simulation。

- **Static ecology**: 所有 options 一开始就存在。
- **Flowing ecology**: options 持续出现。

在 rich, affordable environment 中，用 Thompson sampling 比较：

| Ecology type | Final experienced richness | Late novelty rate | Novelty decay |
|---|---:|---:|---:|
| static | 71.4 | 0.155 | 0.537 |
| flowing | 64.0 | 0.217 | 0.183 |

Interpretation:

Static ecology 产生更高 final stock richness，但 flowing ecology 产生更高 late novelty rate，并且 novelty decay 更小。

核心结论：

**Rich ecology 的关键不只是 options 多，而是 new feasible options 是否持续出现。**

Figure:

- [focused_simulations.png](/home/ubuntu/richness/results/focused_simulations.png)
- [figure5_flowing_arms.png](/home/ubuntu/richness/results/figure5_flowing_arms.png)

## Focused Simulation 2: Exploration Policy Robustness

比较三类 learning policy：

| Policy | Algorithm | Final experienced richness | Late novelty rate | Top-arm concentration |
|---|---|---:|---:|---:|
| low exploration | e-greedy 0.1 | 21.2 | 0.101 | 0.896 |
| adaptive exploration | Thompson | 63.2 | 0.213 | 0.091 |
| forced exploration | e-greedy 0.5 | 97.3 | 0.440 | 0.515 |

Interpretation:

不同 learning policy 产生不同类型的 rich life。

- Low exploration 迅速 routinize，top arm concentration 很高。
- Forced exploration 最大化 novelty，但更像外生强制随机探索。
- Thompson sampling 是最自然的中间模型：Bayesian、reward-sensitive，同时仍能产生 ongoing novelty。

Figure:

- [figure4_algorithm_comparison.png](/home/ubuntu/richness/results/figure4_algorithm_comparison.png)

## Focused Simulation 3: Breadth vs Depth

这个扩展避免模型过度偏向城市和消费选择。现实中的 richness 不只来自尝试新东西，也可能来自深入一个经验。

设置两类 arms：

- **surface arms**: 第一次很新，之后 novelty 快速下降。
- **deep arms**: 重复后 novelty / meaning 慢慢增加。

比较：

- breadth city: K=100，surface arms 多。
- depth town: K=10，deep arms 多。

| Environment | K | Unique arms | Unique/K | Late new-discovery rate | Late richness flow | Cumulative richness |
|---|---:|---:|---:|---:|---:|---:|
| breadth city | 100 | 71.7 | 0.717 | 0.149 | 0.235 | 74.1 |
| depth town | 10 | 10.0 | 0.999 | 0.001 | 0.784 | 134.1 |

Interpretation:

如果 richness 包含 depth，小地方不是自动贫乏。depth town 几乎没有 late new-discovery，但 late richness flow 很高，因为 repeated engagement with deep arms 会持续产生 meaning。

核心结论：

**Richness is not only breadth. It can also come from depth.**

## Robustness Sweep

为了确认 focused conclusions 不是单一设定的产物，我们还做了 broader sensitivity sweep:

- K: `10, 25, 50, 100`
- budget: `tight, moderate, generous`
- cost profile: `low, high`
- algorithm: `e-greedy 0.1`, `e-greedy 0.5`, `Thompson`, `UCB`
- arm regime: `fixed`, `flowing`

关键发现：

| Pattern | Result |
|---|---|
| UCB | 最大化 total conversion，但 late novelty 接近 0 |
| e-greedy 0.5 | 最高 late novelty，但因为强制持续随机探索 |
| Thompson | 最好的理论中间模型，既 adaptive 又保留 ongoing novelty |
| flowing arms | 通常比 fixed arms 更能维持 late novelty |

在 K=100, generous budget, low cost, flowing arms 中：

| Algorithm | Experienced/Ecological | Late discovery rate |
|---|---:|---:|
| e-greedy 0.5 | 0.975 | 0.431 |
| Thompson | 0.628 | 0.209 |
| UCB | 0.999 | 0.000 |

Interpretation:

UCB 把空间探索完，所以 stock conversion 最高，但 flow novelty 消失。e-greedy 0.5 保持高 novelty，但更像人为强制。Thompson 是最适合理论叙事的 learner。

Figures:

- [figure6_sensitivity_late_novelty.png](/home/ubuntu/richness/results/figure6_sensitivity_late_novelty.png)
- [figure7_conversion_vs_late_novelty.png](/home/ubuntu/richness/results/figure7_conversion_vs_late_novelty.png)

## Four Conclusions

### Conclusion 1: Ecological richness does not automatically become experienced richness.

环境里有很多机会，不代表 agent 会真的经历很多。

```text
Ecological opportunity != lived experience
```

### Conclusion 2: The conversion pathway matters.

更准确的模型是：

```text
Ecological richness
-> Feasible richness
-> Experienced richness
```

time, cost, and access 会阻断转换。大城市可能 high ecological richness but low feasible conversion；小地方可能 low ecological richness but high conversion。

### Conclusion 3: Stock richness and flow richness are different.

```text
Stock richness = accumulated diversity
Flow richness = ongoing novelty
```

心理上的 interesting / eventful / dramatic / perspective-changing 可能更接近 flow richness，而不是 final entropy 或 final unique arms。

### Conclusion 4: Rich ecologies matter most when they generate continuing feasible novelty.

最 psychologically rich 的环境不是 options 最多的地方，而是持续产生新机会，且这些机会可进入、可负担、可探索的地方。

Examples:

- campus
- walkable mixed-use neighborhood
- creative community
- research environment
- dense but affordable social ecology

## Final Theoretical Claim

**Psychological richness may emerge from the conversion of ecological possibilities into lived novelty.**

更精确地说：

```text
Ecological possibilities become psychologically rich
only when they become feasible,
then experienced,
then sustained either as stock diversity or ongoing novelty flow.
```

## Suggested Titles

Primary:

**From Ecological Opportunity to Experienced Richness**

Subtitle:

**Why More Possible Lives Do Not Always Produce Richer Lived Lives**

Theory-forward alternative:

**Stock and Flow in Psychological Richness**

## Files

Main scripts:

- [simulate_richness.py](/home/ubuntu/richness/simulate_richness.py)
- [focused_simulations.py](/home/ubuntu/richness/focused_simulations.py)

Main reports:

- [focused_report.md](/home/ubuntu/richness/results/focused_report.md)
- [report.md](/home/ubuntu/richness/results/report.md)

Final report:

- [final_report.md](/home/ubuntu/richness/results/final_report.md)

Key figures:

- [focused_simulations.png](/home/ubuntu/richness/results/focused_simulations.png)
- [richness_layers_thompson.png](/home/ubuntu/richness/results/richness_layers_thompson.png)
- [figure1_cumulative_unique.png](/home/ubuntu/richness/results/figure1_cumulative_unique.png)
- [figure2_new_discoveries.png](/home/ubuntu/richness/results/figure2_new_discoveries.png)
- [figure3_unique_share.png](/home/ubuntu/richness/results/figure3_unique_share.png)
- [figure4_algorithm_comparison.png](/home/ubuntu/richness/results/figure4_algorithm_comparison.png)
- [figure5_flowing_arms.png](/home/ubuntu/richness/results/figure5_flowing_arms.png)
- [figure6_sensitivity_late_novelty.png](/home/ubuntu/richness/results/figure6_sensitivity_late_novelty.png)
- [figure7_conversion_vs_late_novelty.png](/home/ubuntu/richness/results/figure7_conversion_vs_late_novelty.png)

