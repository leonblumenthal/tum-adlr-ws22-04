import gymnasium as gym
import numpy as np


class SpawnFixWrapper(gym.Wrapper):
    """BAS wrapper to reset the env if the env is directly terminated.

    This may happend because the agent spawned inside a boid or the target.
    """

    def reset(self, **kwargs):
        done = True
        while done:
            observation, info = self.env.reset(**kwargs)
            observation, reward, terminated, truncated, info = self.env.step(
                np.zeros(self.env.action_space.shape)
            )
            done = terminated or truncated

        return observation, info
