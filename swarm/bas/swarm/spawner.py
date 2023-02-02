from abc import ABC, abstractmethod

import numpy as np

from swarm.bas.swarm.config import SwarmConfig


class Spawner(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def reset(
        self,
        config: SwarmConfig,
        world_size: np.ndarray,
        np_random: np.random.Generator = np.random,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns the tuple (active_boids_mask, positions, velocities) describing the initial boid states following the spawn policy.

        Args:
            config: Configuration of swarm parameters.
            world_size: World is given by [0, world_size[0]] x [0, world_size[1]]
            np_random: number generator

        """
        pass

    def step(
        self,
        active_boids_mask: np.ndarray,
        positions: np.ndarray,
        velocities: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Updates the arrays from the tuple (active_boids_mask, positions, velocities) of the boid states in-place following the spawn policy.

        Args:
            active_boids_mask: current active_boids_mask.
            positions: boid positions.
            velocities: boid velocities.

        """
        pass


class InstantSpawner(Spawner):
    def __init__(
        self,
        spawn_positions: np.ndarray | None = None,
        spawn_velocities: np.ndarray | None = None,
    ):
        """
        Spawner that implements the following spawn policy: All boids are spawned immediately.
        Args:
            reset_positions: Specific spawn positions. If None, position will be randomly initialized.
            reset_velocities: Specific spawn velocities. If None, velocities will be zero initialized.
        """

        super().__init__()
        self.spawn_positions = spawn_positions
        self.spawn_velocities = spawn_velocities

    def reset(
        self,
        config: SwarmConfig,
        world_size: np.ndarray,
        np_random: np.random.Generator = np.random,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

        assert self.spawn_positions is None or self.spawn_positions.shape == (
            config.num_boids,
            2,
        )
        assert self.spawn_velocities is None or self.spawn_velocities.shape == (
            config.num_boids,
            2,
        )

        # all boids are immediately active
        active_boids_mask = np.full(config.num_boids, True)

        if self.spawn_positions is None:
            # Sample boid positions randomly in world size.
            positions = np_random.uniform(
                low=config.radius,
                high=world_size - config.obstacle_margin,
                size=(config.num_boids, 2),
            )
        else:
            # .copy() is important to prevent modifications of self.spawn_positions in-place.
            positions = self.spawn_positions.copy()

        velocities = np.zeros_like(positions, dtype=float)
        if self.spawn_velocities is not None and config.max_velocity is not None:
            velocities += self.spawn_velocities

        return (active_boids_mask, positions, velocities)


class BernoulliSpawner(Spawner):
    def __init__(
        self,
        spawn_probability: float,
        spawn_radius: float,
        spawn_position: np.ndarray | None = None,
    ):
        """
        Spawner that implements the following spawn policy: At each time step, spawns a boid with probability p in spawn area.
        Args:
            spawn_probability: spawn probability p.
            spawn_radius: Radius of circular spawn area.
            spawn_position: Center position of spawn area. If None, position will be randomly initialized.

        """
        super().__init__()
        self.spawn_probability = spawn_probability
        self.spawn_radius = spawn_radius
        self.spawn_position = spawn_position

    def reset(
        self,
        config: SwarmConfig,
        world_size: np.ndarray,
        np_random: np.random.Generator = np.random,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.config = config
        self.world_size = world_size
        self.np_random = np_random

        if self.spawn_position is None:
            # Sample spawn position randomly in world size.
            self.spawn_position = np_random.uniform(
                low=self.spawn_radius,
                high=world_size - self.spawn_radius,
                size=(1, 2),
            )

        active_boids_mask = np.full(config.num_boids, False)
        positions = np.zeros((config.num_boids, 2), dtype=float)
        velocities = np.zeros_like(positions, dtype=float)
        return (active_boids_mask, positions, velocities)

    def step(
        self,
        active_boids_mask: np.ndarray,
        positions: np.ndarray,
        velocities: np.ndarray,
    ):
        inactive_boids_indices = np.where(active_boids_mask == False)[0]

        if len(inactive_boids_indices) > 0:
            boid_index = inactive_boids_indices[0]
            prob, angle, radius = self.np_random.random(3)
            if prob < self.spawn_probability:
                angle *= 2 * np.pi
                radius *= self.spawn_radius
                active_boids_mask[boid_index] = True

                positions[boid_index] = [
                    radius * np.cos(angle),
                    radius * np.sin(angle),
                ] + self.spawn_position

                velocities[boid_index] = np.zeros((1, 2))
