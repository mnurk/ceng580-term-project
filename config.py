config = {
    "total_tests": 10,
    "max_steps": 20,
    "initial_budget": 30,
    "agent_costs": {
        0: 3,
        1: 2,
        2: 4,
    },
    "v3_reward_weights": {
        "success_reward": 25,
        "progress_reward": 2,
        "error_penalty": 0.5,
        "cost_penalty": 0.2,
        "failure_penalty": 0.2,
        "step_penalty": 0.1,
        "budget_failure_penalty": 5,
    },
}
