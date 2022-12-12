import gymnasium as gym
import numpy as np


class NumNeighborsRewardWrapper(gym.RewardWrapper):
    """BAS wrapper for computing the reward as the number of boids within a specified range around the agent."""

    def __init__(self, env: gym.Env, max_range: float):
        """Initialize the wrapper and set the reward range.

        Args:
            env: (Wrapped) BAS environment.
            max_range: Maximum distance for boids to be considered for the reward.
        """
        super().__init__(env)

        # TODO: This might be problematic because another wrapper also defines this attribute.
        #       Private attributes are the solution but they can not be accessed by the renderers without modififaction.
        self.max_range = max_range

        self._reward_range = (0, env.swarm.num_boids)

    def reward(self, _):
        """Compute reward as number of boids within a specified range around the agent."""
        # TODO: Add fundamental compuatations cache in env.
        distances = np.linalg.norm(
            self.env.swarm.positions - self.env.agent.position, axis=1
        )
        return (distances <= self.max_range).sum()
