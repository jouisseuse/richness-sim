# Ecological Opportunity and Experienced Richness

This repository contains simple bandit simulations exploring when ecological richness becomes experienced richness.

## Core Question

Does having more possible experiences in an environment automatically produce a richer lived experience?

The simulations suggest the answer is no. The useful framework is:

```text
Ecological richness
-> Feasible richness
-> Experienced richness
-> Stock richness / Flow richness
```

- **Ecological richness**: all possible opportunities in the environment.
- **Feasible richness**: opportunities accessible under time and cost constraints.
- **Experienced richness**: opportunities actually chosen by the agent.
- **Stock richness**: accumulated diversity of experience.
- **Flow richness**: ongoing novelty over time.

## Main Files

- `simulate_richness.py`: broader simulation suite, including ecological/feasible/experienced richness, temporal trajectories, algorithm checks, and sensitivity sweeps.
- `focused_simulations.py`: concise three-part simulation suite used for the final story.
- `results/final_report_english.md`: shareable English report.
- `results/focused_report.md`: short focused report.
- `results/report.md`: broader exploratory report.

## Key Outputs

- `results/focused_simulations.png`: compact figure for the three focused simulations.
- `results/richness_layers_thompson.png`: ecological, feasible, and experienced richness.
- `results/figure1_cumulative_unique.png`: cumulative unique experiences over time.
- `results/figure2_new_discoveries.png`: ongoing novelty rate over time.
- `results/figure3_unique_share.png`: unique experiences divided by ecological K.
- `results/figure4_algorithm_comparison.png`: exploration policy comparison.
- `results/figure5_flowing_arms.png`: static versus flowing ecology.
- `results/figure6_sensitivity_late_novelty.png`: sensitivity of late novelty.
- `results/figure7_conversion_vs_late_novelty.png`: stock conversion versus flow novelty.

## Reproducing Results

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the focused simulations:

```bash
python3 focused_simulations.py --outdir results --sims 300 --depth-sims 700
```

Run the broader exploratory simulations:

```bash
python3 simulate_richness.py --outdir results --sims 100 --sensitivity-sims 30
```

## Short Takeaway

Psychological richness may emerge not simply from the number of available opportunities, but from the conversion of ecological possibilities into feasible, lived, and ongoing novelty.
