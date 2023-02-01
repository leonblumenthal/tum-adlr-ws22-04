import gymnasium as gym
import numpy as np
from typing import Generator

from swarm.bas import BASEnv
from swarm.bas.wrappers.observation.components.component import (
    ObservationComponent,
    SharedComputer,
)


class SectionObservationComponent(ObservationComponent):
    """BAS observation component to create the section observation.

    For each radial section/quadrant, compute the normalized
    relative position (- the radius) to the nearest boid respectively.
    """

    def __init__(
        self, num_sections: int, max_range: float, subtract_radius: bool = True
    ):
        self._num_sections = num_sections
        self._max_range = max_range
        self._subtract_radius = subtract_radius

        # Default observation in the center of the sections at the maximum range.
        centered_angles = (
            (np.arange(self._num_sections) + 0.5) * 2 * np.pi / self._num_sections
        )
        self._default_observation = np.array(
            [np.cos(centered_angles), np.sin(centered_angles)]
        ).T

    @property
    def observation_space(self) -> gym.spaces.Box:
        return gym.spaces.Box(-1, 1, (self._num_sections, 2))

    def compute_observation(
        self, env: BASEnv, shared_computer: SharedComputer
    ) -> np.ndarray:

        actual_max_range = self._max_range + self._subtract_radius * env.swarm.radius

        differences = shared_computer.agent_to_boid_differences()
        boid_indices = shared_computer.closest_boid_indices_per_section(
            self._num_sections, actual_max_range
        )

        observation = self._default_observation.copy()
        for section_index, boid_index in enumerate(boid_indices):
            if boid_index is None:
                continue

            observation[section_index] = differences[boid_index] / self._max_range

        return observation


class SectionDistanceObservationComponent(SectionObservationComponent):
    """BAS observation component to create the section distance observation.

    For each radial section/quadrant, compute the normalized
    distance (- the radius) to the nearest boid respectively.
    """

    @property
    def observation_space(self) -> gym.spaces.Box:
        return gym.spaces.Box(0, 1, (self._num_sections,))

    def compute_observation(
        self, env: BASEnv, shared_computer: SharedComputer
    ) -> np.ndarray:
        observation = super().compute_observation(env, shared_computer)

        return np.linalg.norm(observation, axis=1)


class SectionVelocityObservationComponent(ObservationComponent):
    """BAS observation component to create the section velocity observation.

    For each radial section/quadrant, compute the normalized
    (relative) velocity of the nearest boid respectively.
    """

    def __init__(
        self,
        num_sections: int,
        max_range: float,
        relative_to_agent: bool,
        subtract_radius: bool = True,
    ):
        self._num_sections = num_sections
        self._max_range = max_range
        self._relative_to_agent = relative_to_agent
        self._subtract_radius = subtract_radius

    # TODO: Specify based on agent max velocity, swarm max velocity, and relative to agent
    @property
    def observation_space(self) -> gym.spaces.Box:
        return gym.spaces.Box(-1, 1, (self._num_sections, 2))

    def compute_observation(
        self, env: BASEnv, shared_computer: SharedComputer
    ) -> np.ndarray:

        env_to_agent_rotation = shared_computer.env_to_agent_rotation()

        actual_max_range = self._max_range + self._subtract_radius * env.swarm.radius

        boid_indices = shared_computer.closest_boid_indices_per_section(
            self._num_sections, actual_max_range
        )

        observation = np.zeros((self._num_sections, 2))
        for section_index, boid_index in enumerate(boid_indices):
            if boid_index is None:
                continue

            boid_velocity = env.swarm.velocities[boid_index]

            if self._relative_to_agent:
                boid_velocity = boid_velocity - env.agent.velocity

            boid_velocity = env_to_agent_rotation @ boid_velocity

            observation[section_index] = boid_velocity / env.agent.max_velocity

        return observation
