import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev

from env.pipeline_env import PipelineEnv
from env.pipeline_env_v2 import PipelineEnvV2
from experiments.baseline import run_baseline_experiment
from experiments.heuristic_baseline import run_heuristic_experiment
from experiments.random_baseline import run_random_experiment
from experiments.train_q_learning import (
    evaluate_scheduler,
    summarize_results,
    train_q_learning,
)


ENVIRONMENTS = {
    "v1": PipelineEnv,
    "v2": PipelineEnvV2,
}

METRICS = [
    "success_rate",
    "average_reward",
    "average_steps",
    "average_passed_tests",
    "average_error_count",
]


def confidence_interval_95(values):
    if len(values) < 2:
        return 0.0

    return 1.96 * stdev(values) / math.sqrt(len(values))


def summarize_across_runs(rows):
    grouped = {}

    for row in rows:
        key = (row["environment"], row["method"])
        grouped.setdefault(key, []).append(row)

    summary_rows = []

    for (environment, method), method_rows in grouped.items():
        summary = {
            "environment": environment,
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


def run_single_seed(environment_name, env_class, seed, training_episodes, eval_episodes):
    rows = []

    static_summary, _ = run_baseline_experiment(
        num_episodes=eval_episodes,
        env_class=env_class,
        seed=seed,
    )
    rows.append(build_result_row(
        environment_name,
        "static",
        seed,
        training_episodes=0,
        eval_episodes=eval_episodes,
        q_table_size=0,
        summary=static_summary,
    ))

    random_summary, _ = run_random_experiment(
        num_episodes=eval_episodes,
        env_class=env_class,
        seed=seed,
    )
    rows.append(build_result_row(
        environment_name,
        "random",
        seed,
        training_episodes=0,
        eval_episodes=eval_episodes,
        q_table_size=0,
        summary=random_summary,
    ))

    heuristic_summary, _ = run_heuristic_experiment(
        num_episodes=eval_episodes,
        env_class=env_class,
        seed=seed,
    )
    rows.append(build_result_row(
        environment_name,
        "heuristic",
        seed,
        training_episodes=0,
        eval_episodes=eval_episodes,
        q_table_size=0,
        summary=heuristic_summary,
    ))

    scheduler, _ = train_q_learning(
        num_episodes=training_episodes,
        env_class=env_class,
        seed=seed,
    )
    q_learning_results = evaluate_scheduler(
        scheduler=scheduler,
        num_episodes=eval_episodes,
        env_class=env_class,
        seed=seed + 100000,
    )
    q_learning_summary = summarize_results(q_learning_results)
    rows.append(build_result_row(
        environment_name,
        "q_learning",
        seed,
        training_episodes=training_episodes,
        eval_episodes=eval_episodes,
        q_table_size=len(scheduler.q_table),
        summary=q_learning_summary,
    ))

    return rows


def build_result_row(
        environment,
        method,
        seed,
        training_episodes,
        eval_episodes,
        q_table_size,
        summary,
):
    row = {
        "environment": environment,
        "method": method,
        "seed": seed,
        "training_episodes": training_episodes,
        "eval_episodes": eval_episodes,
        "q_table_size": q_table_size,
    }
    row.update({metric: summary[metric] for metric in METRICS})
    return row


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
    training_episodes = 1000
    eval_episodes = 100
    seeds = list(range(30))
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    all_rows = []

    for environment_name, env_class in ENVIRONMENTS.items():
        for seed in seeds:
            seed_rows = run_single_seed(
                environment_name=environment_name,
                env_class=env_class,
                seed=seed,
                training_episodes=training_episodes,
                eval_episodes=eval_episodes,
            )
            all_rows.extend(seed_rows)

    summary_rows = summarize_across_runs(all_rows)

    write_csv(results_dir / "repeated_experiment_runs.csv", all_rows)
    write_csv(results_dir / "repeated_experiment_summary.csv", summary_rows)
    write_json(results_dir / "repeated_experiment_runs.json", all_rows)
    write_json(results_dir / "repeated_experiment_summary.json", summary_rows)

    print("Repeated experiments completed.")
    print(f"Runs written: {len(all_rows)}")
    print(f"Summary rows written: {len(summary_rows)}")
    print(f"Output directory: {results_dir.resolve()}")


if __name__ == "__main__":
    main()
