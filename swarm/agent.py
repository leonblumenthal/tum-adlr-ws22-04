from dataclasses import dataclass

import numpy as np


@dataclass
class AgentConfig:
    """Configurable parameters for an agent."""

    radius: float
    speed: float


class Agent:
    """Agent inside an environment.

    The agent can be steered by providing a direction vector to the `step` method.
    """

    def __init__(
        self,
        config: AgentConfig,
        world_size: np.ndarray,
        np_random: np.random.Generator | None = None,
    ):
        """Initliaze agent.

        This function should generaly be called from inside an environment.

        Args:
            config: Agent configuration parameters.
            world_size: World size of the environemnt.
            np_random: Random number generator, preferably from the Gymanisum environment.
        """

        self.radius = config.radius
        self.speed = config.speed

        # World is [0, world_size[0]] x [0, world_size[1]].
        self.world_size = world_size
        self.np_random = np.random if np_random is None else np_random

    def reset(self):
        """Reset and intialize the position of the agent randomly."""
        self.location = (
            self.np_random.uniform(low=0, high=1, size=2)
            * (self.world_size - 2 * self.radius)
            + self.radius
        )

    def step(self, direction: np.ndarray):
        """Update the agent's position based on the next direction."""
        self.location += direction * self.speed

        # Keep agent in bounds.
        self.location = np.clip(
            self.location,
            a_min=self.radius,
            a_max=self.world_size - self.radius,
        )
