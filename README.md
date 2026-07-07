# Ecological Opportunity and Experienced Richness

This repository contains a compact set of bandit simulations for studying when ecological opportunities become psychologically rich lived experiences.

## Core Idea

More possible experiences do not automatically produce richer lived experience. The useful pathway is:

```text
Ecological richness
-> Feasible richness
-> Experienced richness
-> Stock richness / Flow richness
```

- **Ecological richness**: all possible opportunities in an environment.
- **Feasible richness**: opportunities accessible under time and cost constraints.
- **Experienced richness**: opportunities actually chosen by the agent.
- **Stock richness**: accumulated diversity of experiences.
- **Flow richness**: ongoing novelty over time.

## Repository Contents

- `focused_simulations.py`: main script for the final three focused simulations.
- `simulate_richness.py`: shared simulation utilities and broader exploratory model code.
- `l1_richness.py`: computational-level model that solves the optimal policy exactly and produces richness surfaces over structure and enrichment.
- `campus_l1.py`: computational-level new-campus/new-city activity exploration model.
- `results/final_report_english.md`: concise English report.
- `results/l1_report.md`: L1 computational-level report.
- `results/campus_l1_report.md`: campus/new-city L1 report.
- `results/focused_simulations.png`: summary figure for the three final simulations.
- `results/*_summary.csv`: summary tables for the final simulations.

Large raw simulation traces and intermediate exploratory reports are intentionally excluded.

## Reproduce

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the final focused simulations:

```bash
python3 focused_simulations.py --outdir results --sims 300 --depth-sims 700
```

Run the L1 computational-level surfaces:

```bash
python3 l1_richness.py --outdir results --T 12 --rollouts 1500
```

Run the new-campus/new-city L1 simulation:

```bash
python3 campus_l1.py --outdir results --sims 500
```

## Short Takeaway

Psychological richness may emerge not simply from the number of available opportunities, but from the conversion of ecological possibilities into feasible, lived, and ongoing novelty.
