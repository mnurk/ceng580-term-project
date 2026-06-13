from experiments.baseline import run_baseline_experiment
from experiments.heuristic_baseline import run_heuristic_experiment
from experiments.random_baseline import run_random_experiment
from experiments.train_q_learning import (
    train_q_learning,
    evaluate_scheduler,
    summarize_results
)


def print_summary(name, summary):
    print(f"\n{name}")
    print("-" * len(name))

    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    num_eval_episodes = 100
    num_training_episodes = 1000

    static_summary, _ = run_baseline_experiment(
        num_episodes=num_eval_episodes
    )

    random_summary, _ = run_random_experiment(
        num_episodes=num_eval_episodes
    )

    heuristic_summary, _ = run_heuristic_experiment(
        num_episodes=num_eval_episodes
    )

    scheduler, training_results = train_q_learning(
        num_episodes=num_training_episodes,
        verbose=False
    )

    q_learning_eval_results = evaluate_scheduler(
        scheduler=scheduler,
        num_episodes=num_eval_episodes
    )

    q_learning_summary = summarize_results(q_learning_eval_results)

    print_summary("Static Baseline", static_summary)
    print_summary("Random Baseline", random_summary)
    print_summary("Heuristic Baseline", heuristic_summary)
    print_summary("Q-learning Scheduler", q_learning_summary)

    print("\nTraining episodes:", num_training_episodes)
    print("Evaluation episodes:", num_eval_episodes)
    print("Learned Q-table size:", len(scheduler.q_table))
