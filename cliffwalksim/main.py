import gymnasium as gym
import numpy as np

from agents.agentfactory import AgentFactory
from agents.tabularagent import TabularAgent
from util.metricstracker import MetricsTracker

NUM_EPISODES = 500
NUM_RUNS = 500


def env_interaction(env_str: str, agent_str: str,
                    behaviour_tracker: MetricsTracker,
                    target_tracker: MetricsTracker,
                    visualize_end_run: bool = False) -> None:
    env = gym.make(env_str, render_mode='ansi', max_episode_steps=100)
    agent = AgentFactory.create_agent(agent_str, env)
    obs, info = env.reset()

    def run_episode(use_target: bool):
        obs, info = env.reset()
        episode_return = 0
        while True:
            old_obs = obs
            action = agent.policy(obs, use_target)
            obs, reward, terminated, truncated, info = env.step(action)
            if not use_target:
                agent.update((old_obs, action, reward, obs))
            episode_return += reward

            if terminated or truncated:
                obs, info = env.reset()
                return episode_return

    for episode in range(NUM_EPISODES):
        # print(f"episode {episode + 1}/{NUM_EPISODES}")
        G = run_episode(use_target=False)
        behaviour_tracker.record_return(agent_str, G)

        G = run_episode(use_target=True)
        target_tracker.record_return(agent_str, G)

    # Visualize run
    def visualize_run(env: gym.Env, agent: TabularAgent):
        terminated = False
        truncated = False
        env = gym.make("CliffWalking-v0", render_mode='human')
        obs, info = env.reset()

        if agent_type != "RANDOM":
            agent.pol.epsilon = 0

        while not (terminated or truncated):
            old_obs = obs
            action = agent.policy(obs)
            obs, reward, terminated, truncated, info = env.step(action)

    if visualize_end_run and agent_type != "RANDOM":
        visualize_run(env, agent)
    env.close()


if __name__ == "__main__":
    target_tracker = MetricsTracker()
    behaviour_tracker = MetricsTracker()
    agent_types = ["RANDOM", "SARSA", "Q-LEARNING", "DOUBLE-Q-LEARNING"]

    select_agent = False
    if select_agent:
        agent_type = input(
            "Input agent-type from [RANDOM, SARSA, Q-LEARNING, DOUBLE-Q-LEARNING]: ")
        if agent_type not in agent_types:
            raise ValueError(f"{agent_type} not in {agent_types}")

        agent_types = [agent_type]

    target_data = {"RANDOM": np.zeros((NUM_EPISODES)),
                   "SARSA": np.zeros((NUM_EPISODES)),
                   "Q-LEARNING": np.zeros((NUM_EPISODES)),
                   "DOUBLE-Q-LEARNING": np.zeros((NUM_EPISODES))}
    behaviour_data = {"RANDOM": np.zeros((NUM_EPISODES)),
                      "SARSA": np.zeros((NUM_EPISODES)),
                      "Q-LEARNING": np.zeros((NUM_EPISODES)),
                      "DOUBLE-Q-LEARNING": np.zeros((NUM_EPISODES))}

    for run in range(NUM_RUNS):
        train_t_tracker = MetricsTracker()
        train_b_tracker = MetricsTracker()
        print(f"run {run + 1}/{NUM_RUNS}")
        # Train
        for agent_type in agent_types:
            print("currently running ", agent_type)
            env_interaction("CliffWalking-v0", agent_type, train_b_tracker,
                            train_t_tracker, visualize_end_run=False)
            target_data[agent_type] += \
            train_t_tracker.return_history[agent_type][0]
            behaviour_data[agent_type] += \
            train_b_tracker.return_history[agent_type][0]
        # train_b_tracker.plot()
        # train_t_tracker.plot()


    def plot(data: dict, tracker: MetricsTracker, policy: str):
        for agent_type, returns in data.items():
            data[agent_type] /= NUM_RUNS
            for r in returns:
                tracker.record_return(agent_type, r)
        tracker.plot(x_axis_label="Episode", title=f"{policy} Return History")


    plot(behaviour_data, behaviour_tracker, "Behaviour")
    plot(target_data, target_tracker, "Target")
