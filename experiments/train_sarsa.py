import random

from config import config
from env.pipeline_env import PipelineEnv
from experiments.train_q_learning import evaluate_scheduler, summarize_results
from rl.sarsa_scheduler import SARSAScheduler


def train_sarsa(
        num_episodes=1000,
        verbose=False,
        env_class=PipelineEnv,
        seed=None
):
    if seed is not None:
        random.seed(seed)

    env = env_class(config)

    scheduler = SARSAScheduler(
        actions=[0, 1, 2],
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=0.2
    )

    results = []

    for episode in range(num_episodes):
        state = env.reset()
        action = scheduler.choose_action(state)
        total_reward = 0

        while True:
            next_state, reward, done = env.step(action)
            total_reward += reward

            if done:
                scheduler.update(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    next_action=None,
                    done=True
                )
                break

            next_action = scheduler.choose_action(next_state)
            scheduler.update(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                next_action=next_action,
                done=False
            )

            state = next_state
            action = next_action

        episode_result = {
            "episode": episode,
            "total_reward": total_reward,
            "steps": env.step_count,
            "passed_tests": env.passed_tests,
            "error_count": env.error_count,
            "success": env.passed_tests == env.total_tests
        }

        results.append(episode_result)

        if verbose and episode % 100 == 0:
            print(f"Episode {episode}: {episode_result}")

    return scheduler, results


if __name__ == "__main__":
    scheduler, results = train_sarsa(
        num_episodes=1000,
        verbose=True
    )

    training_summary = summarize_results(results)

    evaluation_results = evaluate_scheduler(
        scheduler=scheduler,
        num_episodes=100
    )

    evaluation_summary = summarize_results(evaluation_results)

    print("\nSARSA Training Summary")
    print("----------------------")

    for key, value in training_summary.items():
        print(f"{key}: {value}")

    print("\nSARSA Evaluation Summary")
    print("------------------------")

    for key, value in evaluation_summary.items():
        print(f"{key}: {value}")

    print("\nLearned Q-table size:", len(scheduler.q_table))
