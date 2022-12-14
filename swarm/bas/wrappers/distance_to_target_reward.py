import gymnasium as gym
import numpy as np


class DistanceToTargetRewardWrapper(gym.RewardWrapper):
    """BAS wrapper to compute reward as 1 - normalized distance between agent and target position."""

    def __init__(self, env: gym.Env, position: np.ndarray):
        """Initialize wrapper and set the reward range.

        Args:
            env: (Wrapped) BAS environment.
            position: Position of the target.
        """
        super().__init__(env)

        self._position = position
        # TODO: Compute actual max distance possible based on target position.
        self._max_possible_distance = np.linalg.norm(self.blueprint.world_size)

        self._reward_range = (0, 1)

    def reward(self, _):
        """Compute reward as 1 - normalized distance between agent and target position."""
        distance = np.linalg.norm(self.agent.position - self._position)
        normalized_distance = distance / self._max_possible_distance
        return 1 - normalized_distance
