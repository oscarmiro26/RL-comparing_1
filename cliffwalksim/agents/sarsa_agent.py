import numpy as np
import gymnasium as gym

from cliffwalksim.agents.tabularagent import TabularAgent


class SARSA(TabularAgent):
    """
    SARSA agent implementation with GLIE, leveraging TabularAgent's Q-table.
    """
    def __init__(
            self,
            state_space: gym.spaces.Discrete,
            action_space: gym.spaces.Discrete,
            learning_rate=0.1,
            discount_rate=0.9,
            epsilon=0.07
    ):
        super().__init__(state_space, action_space, learning_rate, discount_rate)
        self.epsilon = epsilon

    def update(self, trajectory: tuple) -> None:
        """
        Update method for SARSA using the current action and the next action.
        """
        old_state, action, reward, new_state = trajectory
        next_action = self.policy(new_state)
        target = reward + self.discount_rate * self.q_table[new_state][next_action]
        delta = target - self.q_table[old_state][action]
        self.q_table[old_state][action] += self.learning_rate * delta

    def policy(self, state, use_target=False) -> int:
        """
        Epsilon-greedy policy with GLIE. Epsilon decays over time.
        """
        if np.random.rand() < self.epsilon:
            return self.env_action_space.sample()
        return np.argmax(self.q_table[state])
