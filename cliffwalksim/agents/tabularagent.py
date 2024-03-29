from abc import ABC, abstractmethod
import numpy as np
import gymnasium as gym


class TabularAgent(ABC):
    """
    Agent abstract base class.
    """

    def __init__(self, state_space: gym.spaces.Discrete, action_space: gym.spaces.Discrete, learning_rate=0.1,
                 discount_rate=0.9):
        """
        Agent Base Class constructor.
        Assumes discrete gymnasium spaces.
        You may want to make these attributes private.
        :param state_space: state space of gymnasium environment.
        :param action_space: action space of gymnasium environment.
        :param learning_rate: of the underlying algorithm.
        :param discount_rate: discount factor (`gamma`).
        """
        self.q_table = np.zeros([state_space.n, action_space.n])
        self.env_action_space = action_space
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate

    @abstractmethod
    def update(self, trajectory: tuple) -> None:
        """
        Where the update rule is applied
        :param trajectory: (S, A, R, S') (Q-learning), (S, A, R, S', A') (SARSA).
        """
        pass

    @abstractmethod
    def policy(self, state, use_target):
        """
        This is where you would do action selection.
        For epsilon greedy you can opt to make a separate object for epsilon greedy
        action selection and use composition.
        :param state:
        :return an action
        """
        pass
