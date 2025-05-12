import random


class QLearningAgent:
    def __init__(
        self, actions, alpha, gamma, epsilon, is_master_training_mode_for_agent=False
    ):
        self.q_table = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon  # Exploration-exploitation trade-off
        self.actions = actions
        self.is_master_training_mode_for_agent = is_master_training_mode_for_agent

    def get_q_values(self, state):
        # Returns Q-values for a state, initializing if new.
        if state not in self.q_table:
            self.q_table[state] = [0.0 for _ in self.actions]
        return self.q_table[state]

    def choose_action(self, state, make_mistake_prob_for_gameplay=0.0):
        q_values = self.get_q_values(state)

        # Gameplay-specific: AI might make a random mistake
        if (
            not self.is_master_training_mode_for_agent
            and random.random() < make_mistake_prob_for_gameplay
        ):
            return random.choice(self.actions)

        # Epsilon-greedy action selection
        if random.random() < self.epsilon:
            return random.choice(self.actions)  # Explore
        else:  # Exploit (choose best known action)
            max_q = max(q_values)
            # Randomly choose among actions with the highest Q-value to break ties.
            if all(q_val == max_q for q_val in q_values):
                best_actions = [
                    a for a, q_val in zip(self.actions, q_values) if q_val == max_q
                ]
                return random.choice(best_actions)
            best_action_indices = [
                i for i, q_val in enumerate(q_values) if q_val == max_q
            ]
            return self.actions[random.choice(best_action_indices)]

    def update(self, state, action, reward, next_state):
        # Q-learning update rule
        q_vals = self.get_q_values(state)
        next_q_vals = self.get_q_values(next_state)
        action_idx = self.actions.index(action)
        best_next_q = max(next_q_vals)

        # Q(s,a) <- Q(s,a) + alpha * (reward + gamma * max_Q(s') - Q(s,a))
        q_vals[action_idx] += self.alpha * (
            reward + self.gamma * best_next_q - q_vals[action_idx]
        )
