import gymnasium as gym
import numpy as np


class RelativeActionWrapper(gym.ActionWrapper):
    """BAS wrapper for transforming (2,) actions from the agent frame into the world frame."""

    def step(self, action):
        agent_to_env_rotation = np.array(
            [
                [np.cos(self.env.agent.angle), -np.sin(self.env.agent.angle)],
                [np.sin(self.env.agent.angle), np.cos(self.env.agent.angle)],
            ]
        )
        return self.env.step(agent_to_env_rotation @ action)
