from config import config
from env.pipeline_env import PipelineEnv
from experiments.train_q_learning import train_q_learning

ACTION_NAMES = {
    0: "coder",
    1: "tester",
    2: "reviewer"
}


def inspect_policy(num_training_episodes=1000, num_inspection_episodes=5):
    scheduler, _ = train_q_learning(
        num_episodes=num_training_episodes,
        verbose=False
    )

    scheduler.epsilon = 0.0

    env = PipelineEnv(config)

    for episode in range(num_inspection_episodes):
        state = env.reset()
        total_reward = 0
        trajectory = []

        while True:
            action = scheduler.choose_action(state)
            next_state, reward, done = env.step(action)

            trajectory.append({
                "state": state,
                "action": ACTION_NAMES[action],
                "next_state": next_state,
                "reward": reward,
                "done": done
            })

            total_reward += reward
            state = next_state

            if done:
                break

        print(f"\nEpisode {episode}")
        print("-" * 30)
        print("Total reward:", total_reward)
        print("Steps:", env.step_count)
        print("Success:", env.passed_tests == env.total_tests)
        print("Final state:", state)

        print("\nTrajectory:")
        for step_index, item in enumerate(trajectory):
            print(
                f"{step_index + 1}. "
                f"state={item['state']} -> "
                f"action={item['action']} -> "
                f"next_state={item['next_state']} | "
                f"reward={item['reward']} | "
                f"done={item['done']}"
            )


if __name__ == "__main__":
    inspect_policy()
