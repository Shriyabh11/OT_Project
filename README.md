# Adaptive Cost-Aware SMPM

A Python implementation for SMPM experiments across multiple objectives.

## Install

```bash
pip install -r requirements.txt
```

## Run

Run experiments from the repository root:

```bash
python run_objective2.py --quick
python run_objective2.py
python run_objective3.py --quick
python run_objective3.py
python obj3_standalone.py
```

Outputs are written to `results/` as JSON files and `fig/` as PDF figures.

## Files

- `smpm.py`, `run_experiment.py` — Objective 1 baseline and diagonal-quadratic benchmark
- `ml_problem.py` — Elastic-net logistic regression, data sharding, Newton prox, and reference solution
- `smpm_general.py` — SMPM loop and sampling policies
- `experiment_utils.py` — Step-size tuning and evaluation helpers
- `run_objective2.py` — Objective 2 comparisons, ablations, and heterogeneity sweep
- `run_objective3.py` — Objective 3 importance tracking and phase-aware ablation
- `obj3_standalone.py` — Standalone Objective 3 implementation

## Settings

The experiment scripts default to `lambda = 0.01`, `mu = 1e-3`, 3 tuning seeds, 10 evaluation seeds, and step-size multipliers `{0.25, ..., 16}`. Edit the constants near the top of `run_objective2.py` and `run_objective3.py` to change them.
