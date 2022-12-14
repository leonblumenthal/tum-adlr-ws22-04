import gymnasium as gym
import numpy as np
from gymnasium import spaces
from swarm.bas import wrappers


class TargetDirectionAndSectionObservationWrapper(wrappers.SectionObservationWrapper):
    """BAS wrapper to create target direction and section observation."""

    ObservationType = np.ndarray

    def __init__(
        self,
        env: gym.Env,
        num_sections: int,
        max_range: float,
        position: np.ndarray,
        subtract_radius: bool = False,
    ):
        """Initalize the wrapper and set the observation space.

        Args:
            env: (Wrapped) BAS environment.
            num_sections: Number of radial section around the agent.
            max_range: Maximum range to consider boids.
            position: Position of the target.
            subtract_radius: Subtract boid radius from distances.
        """
        super().__init__(env, num_sections, max_range, subtract_radius)

        self._position = position
        self._observation_space = spaces.Box(
            low=-1, high=1, shape=(num_sections + 1, 2)
        )

    def observation(self, _) -> np.ndarray:
        """Create observation as direction to the target"""

        section_observation = super().observation(None)

        difference = self._position - self.agent.position
        distance = np.linalg.norm(difference)
        if distance < 1e-6:
            return np.array([0, 0])
        target_observation = difference / distance

        return np.concatenate([section_observation, [target_observation]])
