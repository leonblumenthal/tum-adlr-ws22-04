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

        self._max_range = max_range

        self._reward_range = (0, env.swarm.num_boids)

    def reward(self, _):
        """Compute reward as number of boids within a specified range around the agent."""
        # TODO: Add fundamental compuatations cache in env.
        distances = np.linalg.norm(
            self.swarm.positions - self.agent.position, axis=1
        )
        return (distances <= self._max_range).sum()
