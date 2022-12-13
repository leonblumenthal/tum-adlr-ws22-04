import gymnasium as gym
import numpy as np
from gymnasium import spaces


class SectionObservationWrapper(gym.ObservationWrapper):
    """BAS wrapper to create section observation.

    For each radial section/quadrant, compute the normalized
    relative position to the nearest boid respectiely.
    """

    ObservationType = np.ndarray

    def __init__(self, env: gym.Env, num_sections: int, max_range: float):
        """Initalize the wrapper and set the observation space.

        Args:
            env: (Wrapped) BAS environment.
            num_sections: Number of radial section around the agent.
            max_range: Maximum range to consider boids.
        """
        super().__init__(env)

        self._num_sections = num_sections
        self._max_range = max_range

        self._observation_space = spaces.Box(
            low=-1, high=1, shape=(num_sections, 2)
        )

    def observation(self, _) -> np.ndarray:
        """Create the section observation.

        For each radial section/quadrant, compute the normalized
        relative position to the nearest boid respectiely.

        If a section is empty or the nearest boid is further away than the max range,
        the default observation is centered on the max range border of the section
        """

        # TODO: Add functionality to compute these fundamental values in the base Env.
        #       Multiple wrappers can use the values then without re-computing.
        # Compute differences and distances to all boids.
        differences = self.swarm.positions - self.agent.position
        distances = np.linalg.norm(differences, axis=1)
        # Assign (indices of) boids into their respective radial sections.
        angles = np.arctan2(differences[:, 1], differences[:, 0]) % (2 * np.pi)
        sections = np.floor_divide(angles, 2 * np.pi / self._num_sections).astype(int)

        indices = np.arange(len(differences))

        # Create observation for each section individually.
        observation = np.zeros((self._num_sections, 2))
        for section in range(self._num_sections):
            section_boid_mask = sections == section

            # Try to find nearest boid in section.
            if section_boid_mask.any():
                boid_index = np.argmin(distances[section_boid_mask])
                boid_index = indices[section_boid_mask][boid_index]

                # Create observation if boid is within max range.
                if distances[boid_index] < self._max_range:
                    observation[section] = differences[boid_index] / self._max_range
                    continue

            # Set default observation (centered on the max range border of the section).
            angle = 2 * np.pi * (section + 0.5) / self._num_sections
            observation[section] = np.array([np.cos(angle), np.sin(angle)])

        return observation
