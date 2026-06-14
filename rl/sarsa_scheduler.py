import random


class SARSAScheduler:
    def __init__(
            self,
            actions,
            learning_rate=0.1,
            discount_factor=0.95,
            epsilon=0.2
    ):
        self.actions = actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.q_table = {}

    def _state_key(self, state):
        return tuple(state)

    def _ensure_state_exists(self, state):
        state_key = self._state_key(state)

        if state_key not in self.q_table:
            self.q_table[state_key] = {
                action: 0.0 for action in self.actions
            }

        return state_key

    def choose_action(self, state):
        state_key = self._ensure_state_exists(state)

        if random.random() < self.epsilon:
            return random.choice(self.actions)

        q_values = self.q_table[state_key]
        max_q = max(q_values.values())

        best_actions = [
            action for action, value in q_values.items()
            if value == max_q
        ]

        return random.choice(best_actions)

    def update(self, state, action, reward, next_state, next_action, done):
        state_key = self._ensure_state_exists(state)
        next_state_key = self._ensure_state_exists(next_state)

        current_q = self.q_table[state_key][action]

        if done:
            target_q = reward
        else:
            next_q = self.q_table[next_state_key][next_action]
            target_q = reward + self.discount_factor * next_q

        self.q_table[state_key][action] = current_q + self.learning_rate * (
                target_q - current_q
        )
