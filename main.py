import random

from config import config
from env.pipeline_env import PipelineEnv

env = PipelineEnv(config)
state = env.reset()

print("Initial state:", state)

for i in range(10):
    action = random.randint(0, 2)
    next_state, reward, done = env.step(action)

    print(f"Step {i + 1}")
    print("Action:", action)
    print("Next state:", next_state)
    print("Reward:", reward)
    print("Done:", done)
    print("-" * 30)

    if done:
        break
