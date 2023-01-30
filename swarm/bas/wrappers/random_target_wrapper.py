import gymnasium as gym
import numpy as np


class RandomTargetWrapper(gym.Wrapper):
    """BAS wrapper to generate a new random target in each reset call."""

    def __init__(self, env: gym.Env):
        """Initialize the wrapper.

        Args:
            env: (Wrapped) BAS environment.
        """
        super().__init__(env)

        self._target = np.array([0, 0])

    # TODO: This might need to be moved inside the step method and check for truncated or terminated.
    def reset(self, **kwargs):
        self._target[:] = self.np_random.random(2) * self.env.blueprint.world_size
        return self.env.reset(**kwargs)

    @property
    def target(self) -> np.ndarray:
        """Target position that is reset in each reset call."""
        return self._target
