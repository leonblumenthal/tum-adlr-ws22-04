import gymnasium as gym
import numpy as np

from swarm.bas import BASEnv
from swarm.bas.wrappers.observation.components.component import (
    ObservationComponent,
    SharedComputer,
)


class AgentVelocityObservationComponent(ObservationComponent):
    """BAS observation component to compute the agent velocity in the agent frame."""

    @property
    def observation_space(self) -> gym.spaces.Box:
        return gym.spaces.Box(-1, 1, (2,))

    def compute_observation(
        self, env: BASEnv, shared_computer: SharedComputer
    ) -> np.ndarray:
        speed = np.linalg.norm(env.agent.velocity)
        return np.array([speed, 0])
