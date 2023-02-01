import gymnasium as gym
import numpy as np


class SwarmWarmUpWrapper(gym.Wrapper):
    """BAS wrapper to step the swarm multiple times in each reset call.

    This should be placed underneath any wrappers that access the swarm.
    """

    def __init__(self, env: gym.Env, num_steps: int | tuple[int, int] = 0):
        """Initialize the wrapper.

        Args:
            env: (Wrapped) BAS environment.
            num_steps: Number or range of warmup steps.
        """
        super().__init__(env)

        self._num_steps = num_steps

    # TODO: This might need to be moved inside the step method and check for truncated or terminated.
    def reset(self, **kwargs):
        """Warmup the swarm in each reset."""
        ret = self.env.reset(**kwargs)

        num_steps = self._num_steps
        if isinstance(num_steps, tuple):
            num_steps = self.np_random.integers(*num_steps)

        for _ in range(num_steps):
            self.swarm.step()

        return ret
