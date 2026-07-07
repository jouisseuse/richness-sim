# Campus L1 Model: New City / New Campus Life Exploration

## Scenario

Imagine you just moved to a new city or started at a new campus. Each week you choose one activity: cafes, museums, walking routes, clubs, research talks, volunteering, sports, language exchange, community events, creative workshops, or familiar routines.

Each activity is a possible experience with exposure, future relevance, depth/learnability, richness potential, cost, risk/ambiguity, and subjective reward.

## Computational-level objective

This is not an online learning agent. The model computes which activity is rational to explore given the environment. The key score is net marginal richness gain:

```text
Net_MRG_k = p_k q_k gamma_k exp(-gamma_k h_k)
            - lambda_s cost_k
            - risk_weight_s risk_k
```

Future relevance is environment-shaped:

```text
p_k(t) = rho * (h_k + eps) / sum_j(h_j + eps)
         + (1 - rho) * 1 / K_t
```

- high rho: recurrent ecology, past exposure predicts future relevance.
- low rho: flowing/open ecology, future opportunities are less tied to past exposure.
- enriched state: low cost and risk penalties.
- depleted state: high cost and risk penalties.

## Simulation 1: Optimal curiosity curve

This simulation plots Net_MRG as a function of exposure h across rho, gamma, and resource state.

Expected qualitative pattern:

- high rho: value shifts toward moderately familiar activities.
- low rho: value is highest for novel/low-exposure activities.
- depleted state: the whole curve shifts downward because cost and ambiguity matter more.

Figure: `results/campus_l1_curiosity_curves.png`

## Simulation 2: Policy-environment fit

Each condition uses 500 simulations over 52 weeks.

| Ecology / resource | Stock richness | Late flow richness | Depth richness | Novel choices | Moderate choices | Routine choices |
|---|---:|---:|---:|---:|---:|---:|
| recurrent / enriched | 10.10 | 0.462 | 0.712 | 0.194 | 0.681 | 0.125 |
| flowing / enriched | 20.67 | 0.531 | 0.603 | 0.398 | 0.603 | 0.000 |
| recurrent / depleted | 4.62 | 0.355 | 0.446 | 0.089 | 0.292 | 0.620 |
| flowing / depleted | 14.51 | 0.409 | 0.472 | 0.279 | 0.389 | 0.332 |

Interpretation:

- Recurrent/enriched ecologies support depth and moderate-familiarity exploration.
- Flowing/enriched ecologies support novelty seeking and higher late flow.
- Depleted states reduce exploration value even when opportunities exist.

Figure: `results/campus_l1_policy_fit.png`

## Simulation 3: Policy-environment mismatch

Two policies were optimized for different ecology assumptions:

- recurrent-trained policy: assumes past exposure predicts future relevance.
- flowing-trained policy: assumes future opportunities are less tied to past exposure.

| Actual ecology | Policy | Late flow richness | Depth richness | Late novelty rate | Top-arm concentration |
|---|---|---:|---:|---:|---:|
| recurrent | recurrent-trained | 0.460 | 0.712 | 0.134 | 0.142 |
| recurrent | flowing-trained | 0.512 | 0.651 | 0.196 | 0.078 |
| flowing | flowing-trained | 0.530 | 0.602 | 0.306 | 0.068 |
| flowing | recurrent-trained | 0.464 | 0.714 | 0.168 | 0.140 |

Interpretation:

- A recurrent-trained policy in a flowing ecology under-explores new opportunities relative to a flowing-trained policy.
- A flowing-trained policy in a recurrent ecology chases novelty and gives up some depth.
- Psychological richness is therefore partly policy-environment fit, not only environmental abundance.

Figure: `results/campus_l1_policy_mismatch.png`

## Main conclusion

The campus/city L1 model reframes richness as adaptive exploration under ecological structure and resource constraints. Richness is highest when the policy fits the ecology: recurrent ecologies reward depth and moderate familiarity; flowing ecologies reward novelty seeking; depleted states gate exploration even when opportunities are objectively present.
