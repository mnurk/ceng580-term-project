from config import config
from env.pipeline_env import PipelineEnv


def run_baseline_episode(verbose=False, env_class=PipelineEnv):
    env = env_class(config)
    state = env.reset()

    action_sequence = [0, 1, 2]  # coder, tester, reviewer
    action_index = 0

    total_reward = 0

    while True:
        action = action_sequence[action_index]
        next_state, reward, done = env.step(action)

        total_reward += reward

        if verbose:
            print("Action:", action)
            print("State:", next_state)
            print("Reward:", reward)
            print("Done:", done)
            print("-" * 30)

        if done:
            break

        action_index = (action_index + 1) % len(action_sequence)

    return {
        "final_state": next_state,
        "total_reward": total_reward,
        "steps": env.step_count,
        "passed_tests": env.passed_tests,
        "error_count": env.error_count,
        "success": env.passed_tests == env.total_tests
    }


def run_baseline_experiment(num_episodes=100, env_class=PipelineEnv):
    results = []

    for _ in range(num_episodes):
        result = run_baseline_episode(
            verbose=False,
            env_class=env_class
        )
        results.append(result)

    success_count = sum(1 for result in results if result["success"])
    total_rewards = sum(result["total_reward"] for result in results)
    total_steps = sum(result["steps"] for result in results)
    total_passed_tests = sum(result["passed_tests"] for result in results)
    total_error_count = sum(result["error_count"] for result in results)

    summary = {
        "num_episodes": num_episodes,
        "success_rate": success_count / num_episodes,
        "average_reward": total_rewards / num_episodes,
        "average_steps": total_steps / num_episodes,
        "average_passed_tests": total_passed_tests / num_episodes,
        "average_error_count": total_error_count / num_episodes
    }

    return summary, results


if __name__ == "__main__":
    print("Single baseline episode:")
    single_result = run_baseline_episode(verbose=True)
    print("Baseline Result:", single_result)

    print("\nBaseline experiment:")
    summary, _ = run_baseline_experiment(num_episodes=100)
    print("Baseline Summary:", summary)
