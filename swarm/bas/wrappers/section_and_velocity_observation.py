import gymnasium as gym
import numpy as np
from gymnasium import spaces
from swarm.bas.wrappers import SectionObservationWrapper


class SectionAndVelocityObservationWrapper(SectionObservationWrapper):
    """BAS wrapper to create section observation + agent velocity observation.

    For each radial section/quadrant, compute the normalized
    relative position to the nearest boid respectively.
    """


    def __init__(self, env: gym.Env, num_sections: int, max_range: float):
        """Initalize the wrapper and set the observation space.

        Args:
            env: (Wrapped) BAS environment.
            num_sections: Number of radial section around the agent.
            max_range: Maximum range to consider boids.
        """
        super().__init__(env, num_sections, max_range)

        self._observation_space = spaces.Box(
            low=-1, high=1, shape=(num_sections + 1, 2)
        )

    def observation(self, obs) -> np.ndarray:
        """Create the section observation + velocity observation."""

        section_observation: np.ndarray = super().observation(obs)
        velocity_observation: np.ndarray = self.agent.velocity/self.agent.max_velocity
        return np.concatenate([section_observation, [velocity_observation]])
