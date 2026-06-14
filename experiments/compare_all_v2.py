from env.pipeline_env_v2 import PipelineEnvV2
from experiments.baseline import run_baseline_experiment
from experiments.heuristic_baseline import run_heuristic_experiment
from experiments.random_baseline import run_random_experiment
from experiments.train_q_learning import (
    train_q_learning,
    evaluate_scheduler,
    summarize_results
)
from experiments.train_sarsa import train_sarsa


def print_summary(name, summary):
    print(f"\n{name}")
    print("-" * len(name))

    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    num_eval_episodes = 100
    num_training_episodes = 1000

    static_summary, _ = run_baseline_experiment(
        num_episodes=num_eval_episodes,
        env_class=PipelineEnvV2
    )

    random_summary, _ = run_random_experiment(
        num_episodes=num_eval_episodes,
        env_class=PipelineEnvV2
    )

    heuristic_summary, _ = run_heuristic_experiment(
        num_episodes=num_eval_episodes,
        env_class=PipelineEnvV2
    )

    scheduler, _ = train_q_learning(
        num_episodes=num_training_episodes,
        verbose=False,
        env_class=PipelineEnvV2
    )

    q_learning_eval_results = evaluate_scheduler(
        scheduler=scheduler,
        num_episodes=num_eval_episodes,
        env_class=PipelineEnvV2
    )

    q_learning_summary = summarize_results(q_learning_eval_results)

    sarsa_scheduler, _ = train_sarsa(
        num_episodes=num_training_episodes,
        verbose=False,
        env_class=PipelineEnvV2
    )

    sarsa_eval_results = evaluate_scheduler(
        scheduler=sarsa_scheduler,
        num_episodes=num_eval_episodes,
        env_class=PipelineEnvV2
    )

    sarsa_summary = summarize_results(sarsa_eval_results)

    print_summary("Static Baseline V2", static_summary)
    print_summary("Random Baseline V2", random_summary)
    print_summary("Heuristic Baseline V2", heuristic_summary)
    print_summary("Q-learning Scheduler V2", q_learning_summary)
    print_summary("SARSA Scheduler V2", sarsa_summary)

    print("\nTraining episodes:", num_training_episodes)
    print("Evaluation episodes:", num_eval_episodes)
    print("Learned Q-table size:", len(scheduler.q_table))
    print("Learned SARSA Q-table size:", len(sarsa_scheduler.q_table))
