import gymnasium as gym
from typing import Callable, Any

from swarm.bas import BASEnv


class InfoAddWrapper(gym.Wrapper):
    """BAS wrapper to add arbitrary data into the info dictionary in each step call."""

    def __init__(self, env: gym.Env, callables: dict[str, Callable[[BASEnv], Any]]):
        """Initialize the wrapper.

        Args:
            env: (Wrapped) BAS environment.
            callables: Arbitrary dictionary from info keys to functions with the BAS env as argument.
        """
        super().__init__(env)

        self._callables = callables

    def step(self, action):
        """Add info based on callables."""
        observation, reward, terminated, truncated, info = self.env.step(action)

        for key, f in self._callables.items():
            info[key] = f(self.env)

        return observation, reward, terminated, truncated, info
