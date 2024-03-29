import gymnasium as gym

from cliffwalksim.agents.randomagent import RandomAgent
from cliffwalksim.agents.tabularagent import TabularAgent

from cliffwalksim.agents.sarsa_agent import SARSA
from cliffwalksim.agents.q_learning_agent import QLearning
from cliffwalksim.agents.double_q_learning_agent import DoubleQLearning


class AgentFactory:
    """
    Naive factory method implementation for
    RL agent creation.
    """
    @staticmethod
    def create_agent(agent_type: str, env: gym.Env) -> TabularAgent:
        """
        Factory method for Agent creation.
        :param env: gymnasium environment.
        :param agent_type: a string key corresponding to the agent.
        :return: an object of type Agent.
        """
        obs_space = env.observation_space
        action_space = env.action_space

        if agent_type == "SARSA":
            return SARSA(obs_space, action_space)
        elif agent_type == "Q-LEARNING":
            return QLearning(obs_space, action_space)
        elif agent_type == "DOUBLE-Q-LEARNING":
            return DoubleQLearning(obs_space, action_space)
        elif agent_type == "RANDOM":
            return RandomAgent(obs_space, action_space)

        raise ValueError("Invalid agent type")
