# Reinforcement Learning-Based Orchestration of Multi-Agent Software Development Pipelines

This repository contains a research prototype for studying reinforcement learning-based scheduling of simplified software development agents. The scheduler observes the current pipeline state and selects which role should act next: coder, tester, or reviewer.

The project focuses on orchestration, not on building a new coding agent. The current implementation uses controlled simulated environments so that scheduling behavior can be analyzed before moving to real LLM-based software engineering agents.

## Project Structure

```text
config.py                         Shared experiment configuration
main.py                           Small random-action environment demo

env/
  pipeline_env.py                 V1 simplified pipeline environment
  pipeline_env_v2.py              V2 role-balanced environment
  pipeline_env_v3.py              V3 cost/difficulty/budget-aware environment

rl/
  q_learning_scheduler.py         Tabular Q-learning scheduler
  sarsa_scheduler.py              Tabular SARSA scheduler

experiments/
  baseline.py                     Static coder-tester-reviewer baseline
  random_baseline.py              Random action baseline
  heuristic_baseline.py           Rule-based heuristic baseline
  train_q_learning.py             Q-learning training/evaluation helpers
  train_sarsa.py                  SARSA training helper
  compare_all.py                  V1 method comparison
  compare_all_v2.py               V2 method comparison
  compare_all_v3.py               V3 method comparison
  run_repeated_experiments.py     Multi-seed experiment runner
  plot_results.py                 Plots repeated experiment results
  run_ablation_study.py           V3 reward ablation runner
  plot_ablation_results.py        Plots ablation results
  inspect_policy.py               Prints learned Q-learning trajectories
```

## Environments

- **V1** is the initial simplified environment. It is useful as a first prototype, but policy inspection showed that the reviewer action could become too dominant.
- **V2** prevents the reviewer from solving the final remaining error, making the coder role more necessary.
- **V3** adds task difficulty, agent invocation costs, a budget limit, stochastic agent success, and configurable reward components.

## Methods

The experiments compare:

- static baseline
- random baseline
- heuristic baseline
- tabular Q-learning
- tabular SARSA

## Setup

Install dependencies:

```powershell
pip install -r requirements.txt
```

If `python` is not available on the system PATH, use the Python executable available in your local environment.

## Quick Checks

Run the basic environment demo:

```powershell
python main.py
```

Run single-environment comparisons:

```powershell
python -m experiments.compare_all
python -m experiments.compare_all_v2
python -m experiments.compare_all_v3
```

Train individual RL schedulers:

```powershell
python -m experiments.train_q_learning
python -m experiments.train_sarsa
```

## Repeated Experiments

Run the main multi-seed experiment:

```powershell
python -m experiments.run_repeated_experiments
```

This evaluates all methods on V1, V2, and V3 using 30 random seeds and writes:

```text
results/repeated_experiment_runs.csv
results/repeated_experiment_runs.json
results/repeated_experiment_summary.csv
results/repeated_experiment_summary.json
```

The summary files include mean, standard deviation, and 95% confidence intervals.

Generate result plots:

```powershell
python -m experiments.plot_results
```

Plots are written to:

```text
results/plots/
```

## Reward Ablation Study

Run V3 reward ablations:

```powershell
python -m experiments.run_ablation_study
```

The ablation variants are:

- full reward
- no error penalty
- no cost penalty
- no failure penalty
- no step penalty
- low success reward
- high success reward

Generate ablation plots:

```powershell
python -m experiments.plot_ablation_results
```

## Results Directory

The `results/` directory is ignored by git because it contains generated experiment outputs. The outputs can be regenerated from the scripts above. Final paper figures or tables can later be copied into a separate versioned artifact directory if needed.

## Current Research Use

This code supports an experimental study of whether learned scheduling policies can improve the orchestration of developer-agent pipelines compared with static, random, and heuristic workflows. It also highlights how environment design affects learned orchestration behavior, especially through the transition from V1 to V2 and V3.
