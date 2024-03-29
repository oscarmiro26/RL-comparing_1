from cliffwalksim.agents.tabularagent import TabularAgent


class RandomAgent(TabularAgent):
    def update(self, trajectory: tuple) -> None:
        pass

    def policy(self, state, use_target):
        return self.env_action_space.sample()
