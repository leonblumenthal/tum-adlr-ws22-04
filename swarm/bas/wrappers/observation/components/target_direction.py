import gymnasium as gym
import numpy as np

from swarm.bas import BASEnv
from swarm.bas.wrappers.observation.components.component import (
    ObservationComponent,
    SharedComputer,
)


class TargetDirectionObservationComponent(ObservationComponent):
    """BAS observation component to compute the direction to a target in the agent frame."""

    def __init__(self, target: np.ndarray):
        self._target = target

    @property
    def observation_space(self) -> gym.spaces.Box:
        return gym.spaces.Box(-1, 1, (2,))

    def compute_observation(
        self, env: BASEnv, shared_computer: SharedComputer
    ) -> np.ndarray:

        difference = self._target - env.agent.position
        distance = np.linalg.norm(difference)

        if distance < 1e-6:
            return np.zeros(2)

        env_to_agent_rotation = shared_computer.env_to_agent_rotation()

        direction = env_to_agent_rotation @ (difference / distance)

        return direction
