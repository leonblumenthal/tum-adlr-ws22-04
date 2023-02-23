from abc import ABC, abstractmethod, abstractproperty

import gymnasium as gym
import numpy as np

from bas.wrappers.observation.shared_computer import SharedComputer


class ObservationComponent(ABC):
    """BAS observation component used in the `ObservationContainerWrapper`."""
    @abstractproperty
    def observation_space(self) -> gym.spaces.Box:
        pass

    @abstractmethod
    def compute_observation(
        self, env: gym.Env, shared_computer: SharedComputer
    ) -> np.ndarray:
        pass
