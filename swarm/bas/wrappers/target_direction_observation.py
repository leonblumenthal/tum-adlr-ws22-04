import gymnasium as gym
import numpy as np
from gymnasium import spaces


class TargetDirectionObservationWrapper(gym.ObservationWrapper):
    """BAS wrapper to create target direction observation."""

    ObservationType = np.ndarray

    def __init__(self, env: gym.Env, position: np.ndarray):
        """Initalize the wrapper and set the observation space.

        Args:
            env: (Wrapped) BAS environment.
            position: Position of the target.
        """
        super().__init__(env)

        self._position = position
        self._observation_space = spaces.Box(low=-1, high=1, shape=(2,))

    def observation(self, _) -> np.ndarray:
        """Create observation as direction to the target"""

        difference = self._position - self.agent.position
        distance = np.linalg.norm(difference)

        if distance < 1e-6:
            return np.array([0, 0])

        observation = difference / distance

        return observation
