import gymnasium as gym
import numpy as np

from swarm.bas.wrappers.observation.components.component import ObservationComponent
from swarm.bas.wrappers.observation.shared_computer import SharedComputer


class ObservationContainerWrapper(gym.ObservationWrapper):
    """BAS wrapper to compute modular observations with components (`ObservationComponent`)."""

    def __init__(self, env: gym.Env, components: list[ObservationComponent]):
        super().__init__(env)

        self._shared_computer = SharedComputer(env)
        self._components = components

        self._observation_space = gym.spaces.Tuple(
            [component.observation_space for component in self._components]
        )

    def observation(self, _) -> list[np.ndarray]:
        """Return list of computed observations per component."""
        # cache in shared computer is only valid for one step and must be cleared manually.
        self._shared_computer.clear_cache()

        return [
            component.compute_observation(self.env, self._shared_computer)
            for component in self._components
        ]
