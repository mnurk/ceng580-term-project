from experiments.baseline import run_baseline_experiment
from experiments.random_baseline import run_random_experiment


def print_summary(name, summary):
    print(f"\n{name}")
    print("-" * len(name))

    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    num_episodes = 100

    static_summary, _ = run_baseline_experiment(num_episodes=num_episodes)
    random_summary, _ = run_random_experiment(num_episodes=num_episodes)

    print_summary("Static Baseline", static_summary)
    print_summary("Random Baseline", random_summary)
