import gymnasium as gym
from gymnasium import spaces
import numpy as np


class FlattenObservationWrapper(gym.ObservationWrapper):
    """Flatten `np.ndarray` observation."""

    def __init__(self, env: gym.Env):
        """Initialize wrapper and set flat observation space."""
        super().__init__(env)

        self._observation_space = spaces.Box(
            low=env.observation_space.low.flatten(),
            high=env.observation_space.high.flatten(),
            shape=(np.prod(env.observation_space.shape),),
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        """Flatten `np.ndarray` observation."""
        return observation.flatten()