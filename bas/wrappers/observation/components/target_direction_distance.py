import gymnasium as gym
import numpy as np

from bas import BASEnv
from bas.wrappers.observation.components.component import (
    ObservationComponent,
    SharedComputer,
)


class TargetDirectionDistanceObservationComponent(ObservationComponent):
    """BAS observation component to compute the direction and distance to a target in the agent frame."""

    def __init__(self, target: np.ndarray, max_distance: float, epsilon: float = 0.1):
        self._target = target
        self._max_distance = max_distance
        self._epsilon = epsilon

    @property
    def observation_space(self) -> gym.spaces.Box:
        return gym.spaces.Box(0, 1, (2,))

    def compute_observation(
        self, env: BASEnv, shared_computer: SharedComputer
    ) -> np.ndarray:

        difference = self._target - env.agent.position
        distance = np.linalg.norm(difference)

        if distance < 1e-6:
            return np.zeros(2, dtype=float)

        difference = shared_computer.env_to_agent_rotation() @ difference

        scaled_direction = difference * (
            (1 - self._epsilon) / self._max_distance + self._epsilon / distance
        )

        return scaled_direction
