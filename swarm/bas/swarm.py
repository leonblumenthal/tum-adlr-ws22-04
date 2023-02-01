import numpy as np
from abc import ABC, abstractmethod


def normalize(vectors: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Normalize row vectors if their norm is >0."""

    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    norms += norms < eps
    return vectors / norms


def limit(vectors: np.ndarray, max_norm: float) -> np.ndarray:
    """Rescale row vectors if their norm is greater than `max_norm`."""

    # Clipping is performed on the lower bound.
    clipped_norms = np.linalg.norm(vectors, axis=-1, keepdims=True).clip(max_norm)
    return vectors * max_norm / clipped_norms


class SwarmConfig:
    def __init__(
        self,
        num_boids: int,
        radius: float,
        max_velocity: float | None,
        max_acceleration: float,
        separation_range: float,
        alignment_range: float,
        cohesion_range: float,
        steering_weights: tuple[float, float, float, float],
        obstacle_margin: float,
        target_position: np.ndarray | None = None,
        target_radius: float = 10,
        target_range: float | None = None,
        target_despawn: bool = False,
    ):
        """
        Represents a specific configuration of swarm parameters.

        Args:
            num_boids: Number of boids.
            radius: Radius of each boid in the world.
            max_velocity: Maximum velocity for each boid. None for static boids.
            max_acceleration: Maximum acceleration for each boid.
            separation_range, alignment_range, cohesion_range: Range in which boids are considered for the respective rule.
            steering_weights: Weighting between the rules separation, alignment, cohesion, targeting.
            obstacle_margin: Minimum distance at which boids are forced away from obstacles.
            target_position: Position of target. If None and targeting weight > 0, position will be randomly initialized. Defaults to None.
            target_radius: Radius of target. Defaults to 10.
            target_range: Range in which target is considered. If None, target is always considered for targeting. Defaults to None.
            target_despawn: If true and a target_position exists, boids will despawn when reaching the target. Defaults to False.
        """
        self.num_boids = num_boids
        self.radius = radius
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration
        self.separation_range = separation_range
        self.alignment_range = alignment_range
        self.cohesion_range = cohesion_range
        self.target_range = target_range
        self.steering_weights = steering_weights
        self.obstacle_margin = obstacle_margin
        self.target_position = target_position
        self.target_radius = target_radius
        self.target_despawn = target_despawn


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

    @abstractmethod
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

    def step(
        self,
        active_boids_mask: np.ndarray,
        boid_positions: np.ndarray,
        boid_velocities: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return


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


class Swarm:
    """Swarm of circular boids inside a BAS environment.

    This class should generally be used from within a `BASEnv`.
    It works similar to a gymnasium environment,
    i.e. some attributes are not "declared" until the `reset` method.

    Important:
    All internal operations, especially the distance computations,
    are implemented as dense matrix operations.
    This may lead to unecessary computational overhead if each boid's neighborhood
    only represents a small fraction of the entire swarm.
    """

    def __init__(
        self,
        config: SwarmConfig,
        spawner: Spawner,
        reset_between_episodes: bool = True,
    ):

        """Initialize the swarm.

        The actual boids are created in the `reset` method.

        Args:
            config: Configuration of swarm parameters
            spawner: Spawner of boids following a spawning policy.
        """
        self.config = config
        self.spawner = spawner

        self.active_boids_mask = None
        self.reset_between_episodes = reset_between_episodes

        # Used to set distance to self further than all ranges.
        self._self_distance = max(
            config.separation_range, config.alignment_range, config.cohesion_range
        )

    @property
    def positions(self):
        return self._positions[self.active_boids_mask]

    @property
    def velocities(self):
        return self._velocities[self.active_boids_mask]

    @property
    def radius(self):
        return self.config.radius

    @property
    def target_radius(self):
        return self.config.target_radius

    @property
    def target_position(self):
        return self._target_position

    @property
    def max_velocity(self):
        return self.config.max_velocity

    @property
    def max_acceleration(self):
        return self.config.max_acceleration

    @property
    def num_boids(self):
        return self.config.num_boids

    def reset(self, world_size: np.ndarray, np_random: np.random.Generator = np.random):
        """Store parameters from `BASEnv` and randomly (re)set boid positions and velocities."""
        # Parameters from BASEnv.
        # World is [0, world_size[0]] x [0, world_size[1]].
        self.world_size = world_size
        self.np_random = np_random

        if self.active_boids_mask is None or self.reset_between_episodes:
            (
                self.active_boids_mask,
                self._positions,
                self._velocities,
            ) = self.spawner.reset(self.config, world_size, np_random)

        weights = self.config.steering_weights
        if type(weights) is tuple and len(weights) > 3 and weights[3] > 0:
            if self.config.target_position is None:
                # Sample target position randomly in world size.
                self._target_position = self.np_random.uniform(
                    low=self.config.target_radius,
                    high=self.world_size - self.config.target_radius,
                    size=(2),
                )
            else:
                self._target_position = self.config.target_position
        else:
            self._target_position = None

    def step(self):
        """Update the position and velocity for each boid.

        Each boid individually follows 4 rules:
        1. Collision avoidance (separation)
        2. Velocity matching (alignment)
        3. Flock centering (cohesion)
        4. Steering towards target (targeting)

        Additionally a world edge avoidance is incorporated.

        The position is updated using the Euler method, i.e. v = v + a -> p = p + v.
        """

        # Boids are static.
        if self.max_velocity is None:
            return

        if self.config.target_despawn:
            self.deactivate_target_boids()

        self.spawner.step(self.active_boids_mask, self._positions, self._velocities)

        if not self.active_boids_mask.any():
            return

        # Compute desired velocities and resulting accelerations.
        desired_velocities = self._compute_desired_velocities() * self.max_velocity
        accelerations = desired_velocities - self._velocities[self.active_boids_mask]
        accelerations = limit(accelerations, self.config.max_acceleration)

        # Update velocities.
        self._velocities[self.active_boids_mask] += accelerations
        self._velocities = limit(self._velocities, self.max_velocity)

        # Add velocities to avoid obstacles.
        self._velocities += self._compute_obstacle_bounce() * self.max_velocity

        # Update position.
        self._positions += self._velocities

    def _compute_target_direction(self, positions: np.ndarray) -> np.ndarray:
        """
        Compute normalized desired velocities as given by movement towards the target.
        """
        if self._target_position is None:
            return np.zeros_like(self._velocities[self.active_boids_mask])
        target_direction = self._target_position - positions
        distances = np.linalg.norm(target_direction, axis=-1, keepdims=True)
        return normalize(
            target_direction
            * (
                distances < self.config.target_range
                if self.config.target_range
                else True
            )
        )

    def _compute_separation(
        self, differences: np.ndarray, distances: np.ndarray
    ) -> np.ndarray:
        """
        Compute normalized desired velocities as given by seperation rule:
        Steer away from neighbors that are within the separation range, i.e. negative mean of weighted (1/distance) difference vectors to neighbors.

        Args:
            differences: pairwise difference vectors of boids
            distances: pairwise distance values of boids (with entries in diagonal >= self.separation_range)
        """
        separation = -np.sum(
            differences / distances**2 * (distances < self.config.separation_range),
            axis=1,
        )
        return normalize(separation)

    def _compute_alignment(
        self, velocities: np.ndarray, distances: np.ndarray
    ) -> np.ndarray:
        """
        Compute normalized desired velocities as given by alignment rule:
        Steer in the same direction as neighbors, i.e. mean/sum of velocities of neighbors.

        Args:
            velocities: velocities of boids
            distances: pairwise distance values of boids (with entries in diagonal >= self.alignment_range)
        """
        alignment = np.sum(
            velocities * (distances < self.config.alignment_range), axis=1
        )
        return normalize(alignment)

    def _compute_cohesion(
        self, differences: np.ndarray, distances: np.ndarray
    ) -> np.ndarray:
        """
        Compute normalized desired velocities as given by cohesion rule:
        Steer towards center of neighbors,  i.e. mean of difference vectors to neighbors.

        Args:
            differences: pairwise difference vectors of boids
            distances: pairwise distance values of boids (with entries in diagonal >= self.cohesion_range)
        """
        cohesion = np.sum(
            differences / distances * (distances < self.config.cohesion_range), axis=1
        )
        return normalize(cohesion)

    def _compute_obstacle_bounce(self) -> np.ndarray:
        """Compute the obstacle avoidance velocities.

        Currently this only includes the world edge.

        These are 1 at the point of touching the obstacle and >1 when intersecting.
        When scaled to maximum velocity, this ensures the desired velocities are overpowered.
        """

        # Distances to 0 edge `xp` and width/height edge `xn`.
        xp = self._positions[:, [0]]
        yp = self._positions[:, [1]]
        xn = self.world_size[0] - xp
        yn = self.world_size[1] - yp

        bounce = np.zeros_like(self._velocities)
        # Positive x direction
        dxp = 1 - (xp - self.config.radius) / (
            self.config.obstacle_margin - self.config.radius
        )
        bounce += np.array([[1, 0]]) * (dxp > 0) * dxp
        # Negative x direction
        dxn = 1 - (xn - self.config.radius) / (
            self.config.obstacle_margin - self.config.radius
        )
        bounce += np.array([[-1, 0]]) * (dxn > 0) * dxn
        # Positive y direction
        dyp = 1 - (yp - self.config.radius) / (
            self.config.obstacle_margin - self.config.radius
        )
        bounce += np.array([[0, 1]]) * (dyp > 0) * dyp
        # Negative y direction
        dyn = 1 - (yn - self.config.radius) / (
            self.config.obstacle_margin - self.config.radius
        )
        bounce += np.array([[0, -1]]) * (dyn > 0) * dyn

        return limit(bounce, 1)

    def _compute_desired_velocities(self) -> np.ndarray:
        """Compute the normalized desired velocities for active boids according to the four main rules and target direction."""

        # Compute pairwise difference vectors and distances.
        # difference[i, j] <=> vector from i to j.
        active_positions = self._positions[self.active_boids_mask].astype(float)
        differences = active_positions[None, :] - active_positions[:, None, :]

        distances = np.linalg.norm(differences, axis=-1, keepdims=True)
        # Ensure that the distance to itself is greater than all ranges so that they are not considered in the subsequent calculations.
        distances += np.eye(distances.shape[0])[..., None] * self._self_distance

        separation = self._compute_separation(differences, distances)
        alignment = self._compute_alignment(
            self._velocities[self.active_boids_mask], distances
        )
        cohesion = self._compute_cohesion(differences, distances)
        target_direction = self._compute_target_direction(active_positions)

        # Compute desired velocities as weighted average.
        ws, wa, wc, wt = self.config.steering_weights
        desired_velocities = (
            ws * separation + wa * alignment + wc * cohesion + wt * target_direction
        ) / (ws + wa + wc + wt)

        return desired_velocities

    def deactivate_target_boids(self):
        """Deactivate boids that reached the target area"""
        if self._target_position is None:
            return
        distance_to_target = np.linalg.norm(
            self._positions - self._target_position, axis=-1, keepdims=True
        ).reshape(self.config.num_boids)
        reached_target_indices = np.where(
            distance_to_target < self.config.target_radius
        )
        self.active_boids_mask[reached_target_indices] = False
