import numpy as np
import gymnasium as gym
from cliffwalksim.agents.tabularagent import TabularAgent


class QLearning(TabularAgent):
    """
    Q-learning agent implementation.
    """
    def __init__(
            self,
            state_space: gym.spaces.Discrete,
            action_space: gym.spaces.Discrete,
            learning_rate=0.1,
            discount_rate=0.9,
            epsilon=0.07
    ):
        """
        Q-learning agent constructor.
        :param state_space: state space of gymnasium environment.
        :param action_space: action space of gymnasium environment.
        :param learning_rate: learning rate of the Q-learning algorithm.
        :param discount_rate: discount factor (`gamma`).
        :param epsilon: exploration rate for epsilon-greedy action selection.
        """
        super().__init__(state_space, action_space, learning_rate, discount_rate)
        self.epsilon = epsilon

    def update(self, trajectory: tuple) -> None:
        """
        Q-learning update rule.
        :param trajectory: (S, A, R, S').
        """
        old_state, action, reward, new_state = trajectory

        max_q_new_state = np.max(self.q_table[new_state])
        target = reward + self.discount_rate * max_q_new_state
        delta = target - self.q_table[old_state][action]
        self.q_table[old_state][action] += self.learning_rate * delta

    def policy(self, state, use_target):
        """
        Epsilon-greedy policy.
        :param state: current state.
        :return: action.
        """
        if not use_target and np.random.rand() < self.epsilon:
            return np.random.choice(self.env_action_space.n)

        return np.argmax(self.q_table[state])
