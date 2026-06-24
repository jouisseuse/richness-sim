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
- `results/final_report_english.md`: concise English report.
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

## Short Takeaway

Psychological richness may emerge not simply from the number of available opportunities, but from the conversion of ecological possibilities into feasible, lived, and ongoing novelty.
