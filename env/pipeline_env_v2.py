class PipelineEnvV2:
    def __init__(self, config):
        self.total_tests = config["total_tests"]
        self.max_steps = config["max_steps"]

    def reset(self):
        import random

        self.error_count = random.randint(3, 6)
        self.passed_tests = 0
        self.step_count = 0
        self.last_agent = -1

        return self.__get_state()

    def step(self, action):
        import random

        old_passed_tests = self.passed_tests

        if action == 0:
            # coder
            if random.random() < 0.6:
                self.error_count = max(0, self.error_count - 1)
            else:
                self.error_count += 1

        elif action == 1:
            # tester
            self.passed_tests = max(0, self.total_tests - self.error_count)

        elif action == 2:
            # reviewer
            if self.error_count > 1:
                self.error_count = max(1, self.error_count - random.randint(1, 2))

        else:
            raise ValueError(f"Invalid action: {action}")

        self.step_count += 1
        self.last_agent = action

        reward = 0

        if self.passed_tests == self.total_tests:
            reward += 20

        reward += (self.passed_tests - old_passed_tests) * 2
        reward -= self.error_count * 0.5
        reward -= 0.1

        done = False

        if self.passed_tests == self.total_tests:
            done = True

        if self.step_count >= self.max_steps:
            done = True

        new_state = self.__get_state()
        return new_state, reward, done

    def __get_state(self):
        return [
            self.passed_tests,
            self.error_count,
            self.last_agent,
            self.step_count
        ]
