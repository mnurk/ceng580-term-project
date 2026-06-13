import random

from config import config
from env.pipeline_env import PipelineEnv
from rl.q_learning_scheduler import QLearningScheduler


def train_q_learning(
        num_episodes=1000,
        verbose=False,
        env_class=PipelineEnv,
        seed=None
):
    if seed is not None:
        random.seed(seed)

    env = env_class(config)

    scheduler = QLearningScheduler(
        actions=[0, 1, 2],
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=0.2
    )

    results = []

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0

        while True:
            action = scheduler.choose_action(state)
            next_state, reward, done = env.step(action)

            scheduler.update(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done
            )

            total_reward += reward
            state = next_state

            if done:
                break

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


def evaluate_scheduler(
        scheduler,
        num_episodes=100,
        env_class=PipelineEnv,
        seed=None
):
    if seed is not None:
        random.seed(seed)

    original_epsilon = scheduler.epsilon
    scheduler.epsilon = 0.0

    env = env_class(config)
    results = []

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0

        while True:
            action = scheduler.choose_action(state)
            next_state, reward, done = env.step(action)

            total_reward += reward
            state = next_state

            if done:
                break

        results.append({
            "episode": episode,
            "total_reward": total_reward,
            "steps": env.step_count,
            "passed_tests": env.passed_tests,
            "error_count": env.error_count,
            "success": env.passed_tests == env.total_tests
        })

    scheduler.epsilon = original_epsilon

    return results


def summarize_results(results):
    num_episodes = len(results)

    success_count = sum(1 for result in results if result["success"])
    total_rewards = sum(result["total_reward"] for result in results)
    total_steps = sum(result["steps"] for result in results)
    total_passed_tests = sum(result["passed_tests"] for result in results)
    total_error_count = sum(result["error_count"] for result in results)

    return {
        "num_episodes": num_episodes,
        "success_rate": success_count / num_episodes,
        "average_reward": total_rewards / num_episodes,
        "average_steps": total_steps / num_episodes,
        "average_passed_tests": total_passed_tests / num_episodes,
        "average_error_count": total_error_count / num_episodes
    }


if __name__ == "__main__":
    scheduler, results = train_q_learning(
        num_episodes=1000,
        verbose=True
    )

    training_summary = summarize_results(results)

    evaluation_results = evaluate_scheduler(
        scheduler=scheduler,
        num_episodes=100
    )

    evaluation_summary = summarize_results(evaluation_results)

    print("\nQ-learning Training Summary")
    print("---------------------------")

    for key, value in training_summary.items():
        print(f"{key}: {value}")

    print("\nQ-learning Evaluation Summary")
    print("-----------------------------")

    for key, value in evaluation_summary.items():
        print(f"{key}: {value}")

    print("\nLearned Q-table size:", len(scheduler.q_table))
