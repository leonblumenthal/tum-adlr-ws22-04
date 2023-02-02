import functools
import gymnasium as gym
import numpy as np


class SharedComputer:
    """Shared instance to compute BAS environment values in different components.

    This should be recreated each step to "delete" the cache.
    """

    def __init__(self, env: gym.Env) -> None:
        self._env = env

    @functools.cache
    def env_to_agent_rotation(self) -> np.ndarray:
        return np.array(
            [
                [np.cos(self._env.agent.angle), np.sin(self._env.agent.angle)],
                [-np.sin(self._env.agent.angle), np.cos(self._env.agent.angle)],
            ]
        )

    @functools.cache
    def agent_to_env_rotation(self) -> np.ndarray:
        return self.env_to_agent_rotation().T

    @functools.cache
    def agent_to_boid_differences(
        self, in_agent_coordinate_frame: bool = True
    ) -> np.ndarray:
        differences = self._env.swarm.positions - self._env.agent.position
        if in_agent_coordinate_frame:
            differences = differences @ self.env_to_agent_rotation().T
        return differences

    @functools.cache
    def agent_to_boid_distances(
        self, in_agent_coordinate_frame: bool = True
    ) -> np.ndarray:
        differences = self.agent_to_boid_differences(in_agent_coordinate_frame)
        distances = np.linalg.norm(differences, axis=1)
        return distances

    @functools.cache
    def agent_to_boid_angles(
        self, in_agent_coordinate_frame: bool = True
    ) -> np.ndarray:
        differences = self.agent_to_boid_differences(in_agent_coordinate_frame)
        angles = np.arctan2(differences[:, 1], differences[:, 0]) % (2 * np.pi)
        return angles

    @functools.cache
    def boid_section_indices(
        self, num_sections: int, in_agent_coordinate_frame: bool = True
    ) -> np.ndarray:
        angles = self.agent_to_boid_angles(in_agent_coordinate_frame)
        indices = np.floor_divide(angles, 2 * np.pi / num_sections).astype(int)
        return indices

    @functools.cache
    def closest_boid_indices_per_section(
        self,
        num_sections: int,
        max_range: float,
        in_agent_coordinate_frame: bool = True,
    ) -> list[int | None]:
        distances = self.agent_to_boid_distances(in_agent_coordinate_frame)
        section_indices = self.boid_section_indices(
            num_sections, in_agent_coordinate_frame
        )

        indices = [None] * num_sections

        boid_indices = np.arange(len(distances))

        for section_index in range(num_sections):
            section_mask = section_indices == section_index

            if not section_mask.any():
                continue

            closes_boid_index = boid_indices[section_mask][
                np.argmin(distances[section_mask])
            ]

            if distances[closes_boid_index] > max_range:
                continue

            indices[section_index] = closes_boid_index

        return indices

    def clear_cache(self):
        self.env_to_agent_rotation.cache_clear()
        self.agent_to_env_rotation.cache_clear()
        self.agent_to_boid_differences.cache_clear()
        self.agent_to_boid_distances.cache_clear()
        self.agent_to_boid_angles.cache_clear()
        self.boid_section_indices.cache_clear()
        self.closest_boid_indices_per_section.cache_clear()