import csv
import json
import math
from copy import deepcopy
from pathlib import Path
from statistics import mean, stdev

from config import config
from env.pipeline_env_v3 import PipelineEnvV3
from experiments.heuristic_baseline import run_heuristic_experiment
from experiments.train_q_learning import evaluate_scheduler, summarize_results, train_q_learning
from experiments.train_sarsa import train_sarsa


BASE_REWARD_WEIGHTS = config["v3_reward_weights"]
ABLATIONS = {
    "full_reward": {},
    "no_error_penalty": {"error_penalty": 0.0},
    "no_cost_penalty": {"cost_penalty": 0.0},
    "no_failure_penalty": {"failure_penalty": 0.0},
    "no_step_penalty": {"step_penalty": 0.0},
    "low_success_reward": {"success_reward": 10},
    "high_success_reward": {"success_reward": 40},
}

METHODS = ["heuristic", "q_learning", "sarsa"]
METRICS = [
    "success_rate",
    "average_reward",
    "average_steps",
    "average_passed_tests",
    "average_error_count",
]


def build_config(overrides):
    ablation_config = deepcopy(config)
    reward_weights = deepcopy(BASE_REWARD_WEIGHTS)
    reward_weights.update(overrides)
    ablation_config["v3_reward_weights"] = reward_weights
    return ablation_config


def make_env_class(ablation_config):
    class AblationPipelineEnvV3(PipelineEnvV3):
        def __init__(self, _):
            super().__init__(ablation_config)

    return AblationPipelineEnvV3


def confidence_interval_95(values):
    if len(values) < 2:
        return 0.0

    return 1.96 * stdev(values) / math.sqrt(len(values))


def build_result_row(ablation, method, seed, training_episodes, eval_episodes, q_table_size, summary):
    row = {
        "ablation": ablation,
        "method": method,
        "seed": seed,
        "training_episodes": training_episodes,
        "eval_episodes": eval_episodes,
        "q_table_size": q_table_size,
    }
    row.update({metric: summary[metric] for metric in METRICS})
    return row


def summarize_across_runs(rows):
    grouped = {}

    for row in rows:
        key = (row["ablation"], row["method"])
        grouped.setdefault(key, []).append(row)

    summary_rows = []

    for (ablation, method), method_rows in grouped.items():
        summary = {
            "ablation": ablation,
            "method": method,
            "num_runs": len(method_rows),
            "eval_episodes": method_rows[0]["eval_episodes"],
            "training_episodes": method_rows[0]["training_episodes"],
        }

        for metric in METRICS:
            values = [row[metric] for row in method_rows]
            summary[f"{metric}_mean"] = mean(values)
            summary[f"{metric}_std"] = stdev(values) if len(values) > 1 else 0.0
            summary[f"{metric}_ci95"] = confidence_interval_95(values)

        summary_rows.append(summary)

    return summary_rows


def run_single_seed(ablation_name, env_class, seed, training_episodes, eval_episodes):
    rows = []

    heuristic_summary, _ = run_heuristic_experiment(
        num_episodes=eval_episodes,
        env_class=env_class,
        seed=seed,
    )
    rows.append(build_result_row(
        ablation_name,
        "heuristic",
        seed,
        training_episodes=0,
        eval_episodes=eval_episodes,
        q_table_size=0,
        summary=heuristic_summary,
    ))

    q_scheduler, _ = train_q_learning(
        num_episodes=training_episodes,
        env_class=env_class,
        seed=seed,
    )
    q_results = evaluate_scheduler(
        scheduler=q_scheduler,
        num_episodes=eval_episodes,
        env_class=env_class,
        seed=seed + 100000,
    )
    rows.append(build_result_row(
        ablation_name,
        "q_learning",
        seed,
        training_episodes=training_episodes,
        eval_episodes=eval_episodes,
        q_table_size=len(q_scheduler.q_table),
        summary=summarize_results(q_results),
    ))

    sarsa_scheduler, _ = train_sarsa(
        num_episodes=training_episodes,
        env_class=env_class,
        seed=seed,
    )
    sarsa_results = evaluate_scheduler(
        scheduler=sarsa_scheduler,
        num_episodes=eval_episodes,
        env_class=env_class,
        seed=seed + 100000,
    )
    rows.append(build_result_row(
        ablation_name,
        "sarsa",
        seed,
        training_episodes=training_episodes,
        eval_episodes=eval_episodes,
        q_table_size=len(sarsa_scheduler.q_table),
        summary=summarize_results(sarsa_results),
    ))

    return rows


def write_csv(path, rows):
    if not rows:
        return

    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, rows):
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(rows, output_file, indent=2)


def main():
    training_episodes = 10000
    eval_episodes = 100
    seeds = list(range(30))
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    all_rows = []

    for ablation_name, overrides in ABLATIONS.items():
        ablation_config = build_config(overrides)
        env_class = make_env_class(ablation_config)

        for seed in seeds:
            rows = run_single_seed(
                ablation_name=ablation_name,
                env_class=env_class,
                seed=seed,
                training_episodes=training_episodes,
                eval_episodes=eval_episodes,
            )
            all_rows.extend(rows)

    summary_rows = summarize_across_runs(all_rows)

    write_csv(results_dir / "ablation_runs.csv", all_rows)
    write_csv(results_dir / "ablation_summary.csv", summary_rows)
    write_json(results_dir / "ablation_runs.json", all_rows)
    write_json(results_dir / "ablation_summary.json", summary_rows)

    print("Ablation study completed.")
    print(f"Runs written: {len(all_rows)}")
    print(f"Summary rows written: {len(summary_rows)}")
    print(f"Output directory: {results_dir.resolve()}")


if __name__ == "__main__":
    main()
