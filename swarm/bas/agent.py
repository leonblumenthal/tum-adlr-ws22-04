import numpy as np
from gymnasium import spaces


class Agent:
    """Circular agent inside insdie a BAS environment.

    This class should generally be used from within a `BASEnv`.
    It works similar to a gymnasium environment,
    i.e. some attributes are not "declared" until the `reset` method.

    It accepts a normalized direction as action,
    which is scaled to `max_velocity` for the next velocity.

    In the `reset` method, the agent either spawns randomly in the world
    or at `reset_position` if specified.
    """

    ActionType = np.ndarray

    action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=float)

    def __init__(
        self,
        radius: float,
        max_velocity: float,
        max_acceleration: float,
        reset_position: np.ndarray | None = None,
    ):
        """Initilaize the agent.

        The actual position is created in the `reset` method.

        Args:
            radius: Radius of the agent in the world
            max_velocity: Maximum velocity of the agent.
            max_acceleration: Maximum acceleration of the agent.
            reset_position: Specific spawn position. Defaults to None.
        """
        self.radius = radius
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration
        self.reset_position = reset_position

        #initialized here to allow referencing before reset is called
        self.position = np.zeros(2,dtype=float)

    def reset(self, world_size: np.ndarray, np_random: np.random.Generator = np.random):
        """Store parameters from `BASEnv` and
        (re)set the position randomly or to `reset_position` if specified.
        """
        # Parameters from BASEnv.
        # World is [0, world_size[0]] x [0, world_size[1]].
        self.world_size = world_size
        self.np_random = np_random

        if self.reset_position is None:
            # Set position randomly without intersecting the world border.
            self.position[:] = (
                self.np_random.uniform(low=0, high=1, size=2)
                * (self.world_size - 2 * self.radius)
                + self.radius
            )
        else:
            # .copy() is important because self.position is modified in-place in self.step().
            self.position[:] = self.reset_position.astype(float).copy()

        self.velocity = np.zeros_like(self.position)
        self.angle = 0

    def step(self, desired_velocity: ActionType):
        """Update the position with the next velocity based desired velocity"""

        acceleration = desired_velocity - self.velocity
        clipped_acc_norm = np.linalg.norm(acceleration).clip(self.max_acceleration)
        acceleration = acceleration * self.max_acceleration / clipped_acc_norm

        self.velocity += acceleration
        clipped_vel_norm = np.linalg.norm(self.velocity).clip(self.max_velocity)
        self.velocity = self.velocity * self.max_velocity / clipped_vel_norm
        self.angle = np.arctan2(self.velocity[1], self.velocity[0])

        self.position += self.velocity

        # Keep agent in bounds.
        self.position[:] = np.clip(
            self.position,
            a_min=self.radius,
            a_max=self.world_size - self.radius,
        )
