import random


class PipelineEnvV3:
    def __init__(self, config):
        self.total_tests = config["total_tests"]
        self.max_steps = config["max_steps"]
        self.initial_budget = config.get("initial_budget", 30)
        self.agent_costs = config.get(
            "agent_costs",
            {
                0: 3,  # coder
                1: 2,  # tester
                2: 4,  # reviewer
            }
        )
        self.reward_weights = config.get(
            "v3_reward_weights",
            {
                "success_reward": 25,
                "progress_reward": 2,
                "error_penalty": 0.5,
                "cost_penalty": 0.2,
                "failure_penalty": 0.2,
                "step_penalty": 0.1,
                "budget_failure_penalty": 5,
            }
        )

    def reset(self):
        self.difficulty = random.randint(0, 2)
        self.error_count = random.randint(
            2 + self.difficulty * 2,
            3 + self.difficulty * 2
        )
        self.passed_tests = 0
        self.step_count = 0
        self.last_agent = -1
        self.budget_remaining = self.initial_budget
        self.consecutive_failures = 0

        return self.__get_state()

    def step(self, action):
        old_passed_tests = self.passed_tests
        old_error_count = self.error_count

        action_cost = self.agent_costs[action]
        self.budget_remaining -= action_cost

        if action == 0:
            self.__run_coder()
        elif action == 1:
            self.__run_tester()
        elif action == 2:
            self.__run_reviewer()
        else:
            raise ValueError(f"Invalid action: {action}")

        self.step_count += 1
        self.last_agent = action

        if self.error_count < old_error_count or self.passed_tests > old_passed_tests:
            self.consecutive_failures = 0
        else:
            self.consecutive_failures += 1

        reward = self.__calculate_reward(
            old_passed_tests=old_passed_tests,
            action_cost=action_cost
        )
        done = self.__is_done()

        if self.budget_remaining <= 0 and self.passed_tests < self.total_tests:
            reward -= self.reward_weights["budget_failure_penalty"]

        return self.__get_state(), reward, done

    def __run_coder(self):
        success_probabilities = [0.75, 0.6, 0.45]
        success_probability = success_probabilities[self.difficulty]

        if random.random() < success_probability:
            reduction = 1
            if self.difficulty == 0 and random.random() < 0.25:
                reduction = 2
            self.error_count = max(0, self.error_count - reduction)
        else:
            self.error_count += 1

    def __run_tester(self):
        self.passed_tests = max(0, self.total_tests - self.error_count)

    def __run_reviewer(self):
        if self.error_count <= 1:
            return

        find_probabilities = [0.8, 0.65, 0.5]
        find_probability = find_probabilities[self.difficulty]

        if random.random() < find_probability:
            reduction = random.randint(1, 2)
            self.error_count = max(1, self.error_count - reduction)

    def __calculate_reward(self, old_passed_tests, action_cost):
        reward = 0

        if self.passed_tests == self.total_tests:
            reward += self.reward_weights["success_reward"]

        reward += (
                (self.passed_tests - old_passed_tests)
                * self.reward_weights["progress_reward"]
        )
        reward -= self.error_count * self.reward_weights["error_penalty"]
        reward -= action_cost * self.reward_weights["cost_penalty"]
        reward -= self.consecutive_failures * self.reward_weights["failure_penalty"]
        reward -= self.reward_weights["step_penalty"]

        return reward

    def __is_done(self):
        if self.passed_tests == self.total_tests:
            return True

        if self.step_count >= self.max_steps:
            return True

        return self.budget_remaining <= 0

    def __get_state(self):
        return [
            self.passed_tests,
            self.error_count,
            self.last_agent,
            self.step_count,
            self.difficulty,
            self.budget_remaining,
            self.consecutive_failures
        ]
