import gymnasium as gym
import numpy as np


class DistanceToTargetRewardWrapper(gym.RewardWrapper):
    """BAS wrapper to compute reward as negative distance between agent and target position."""

    def __init__(self, env: gym.Env, target_position: np.ndarray):
        """Initialize wrapper and set the reward range.

        Args:
            env: (Wrapped) BAS environment.
            target_position: Position of the target.
        """
        super().__init__(env)

        self.target_position = target_position

        # TODO: Compute actual max distance possible.
        self._reward_range = (-np.linalg.norm(self.env.blueprint.world_size), 0)

    def reward(self, _):
        """Compute reward as negative distance between agent and target position."""

        return -np.linalg.norm(self.env.agent.position - self.target_position)
