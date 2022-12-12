import numpy as np


def normalize(vectors: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Normalize row vectors if their norm is >0."""

    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    norms += norms < eps
    return vectors / norms


def limit(vectors: np.ndarray, max_norm: float) -> np.ndarray:
    """Rescale row vectors if their norm is greater than `max_norm`."""

    clipped_norms = np.linalg.norm(vectors, axis=-1, keepdims=True).clip(max_norm)
    return vectors * max_norm / clipped_norms


# TODO: Improve modularity.
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
        num_boids: int,
        radius: int,
        max_velocity: int | None,
        max_acceleration: int,
        separation_range: float,
        alignment_range: float,
        cohesion_range: float,
        steering_weights: tuple[float, float, float],
        obstacle_margin: float,
        reset_positions: np.ndarray | None = None,
        reset_velocities: np.ndarray | None = None,
    ):
        """Initialize the swarm.

        The actual boids are created in the `reset` method.

        Args:
            num_boids: Number of boids.
            radius: Radius of each beach in the world.
            max_velocity: Maximum velocity for each boid. None for static boids.
            max_acceleration: Maximum acceleration for each boid.
            separation_range, alignment_range, cohesion_range: Range in which boids are considered for the respective rule.
            steering_weights: Weighting between the three rules mentioned above.
            obstacle_margin: Minimum distance at which boids are force awa from obstacles.
            reset_positions: Specific spawn positions. Defaults to None.
            reset_velocities: Specific spawn velocities. Defaults to None.
        """

        self.num_boids = num_boids
        self.radius = radius
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration
        self.separation_range = separation_range
        self.alignment_range = alignment_range
        self.cohesion_range = cohesion_range
        self.steering_weights = steering_weights
        self.obstacle_margin = obstacle_margin

        assert reset_positions is None or reset_positions.shape == (num_boids, 2)
        self.reset_positions = reset_positions
        assert reset_velocities is None or reset_velocities.shape == (num_boids, 2)
        self.reset_velocities = reset_velocities

        # Used to set distance to self further than all ranges.
        self._self_distance = max(separation_range, alignment_range, cohesion_range)

    def reset(self, world_size: np.ndarray, np_random: np.random.Generator = np.random):
        """Store parameters from `BASEnv` and randomly (re)set boid positions and velocities."""
        # Parameters from BASEnv.
        # World is [0, world_size[0]] x [0, world_size[1]].
        self.world_size = world_size
        self.np_random = np_random

        if self.reset_positions is None:
            # Sample boid posisition randomly in world size.
            self.positions = self.np_random.uniform(
                low=self.radius,
                high=self.world_size - self.obstacle_margin,
                size=(self.num_boids, 2),
            )
        else:
            # .copy() is important because self.positions is modified in-place in self.step().
            self.positions = self.reset_positions.copy()
        
        # Velocities from last step are needed for computation.
        self.velocities = np.zeros_like(self.positions)
        if self.reset_velocities is not None and self.max_velocity is not None:
            self.velocities += self.reset_velocities

    def step(self):
        """Update the position and velocity for each boid.

        Each boid individually follows 3 rules:
        1. Collision avoidance (separation)
        2. Velocity matching (alignment)
        3. Flock centering (cohesion)

        Additionally and world edge avoidance is incorporated.

        The position is updated using the Euler method, i.e. v = v + a -> p = p + v.
        """

        # Boids are static.
        if self.max_velocity is None:
            return

        # Compute desired velocities and resulting accelerations.
        desired_velocities = self._compute_desired_velocities() * self.max_velocity
        accelerations = desired_velocities - self.velocities
        accelerations = limit(accelerations, self.max_acceleration)

        # Update velocities.
        self.velocities += accelerations
        self.velocities = limit(self.velocities, self.max_velocity)

        # Add velocities to avoid obstacles.
        self.velocities += self._compute_obstacle_bounce() * self.max_velocity

        # Update position.
        self.positions += self.velocities

    def _compute_desired_velocities(self) -> np.ndarray:
        """Compute the normalized desired velocities according to the three main rules."""

        # Compute pairwise difference vectors and distances.
        # difference[i, j] <=> vector from i to j.
        differences = self.positions[None, :] - self.positions[:, None, :]
        distances = np.linalg.norm(differences, axis=-1, keepdims=True)
        # Ensure that the distance to itself is greater than all ranges.
        distances += np.eye(distances.shape[0])[..., None] * self._self_distance

        # 1. Separation: Steer away from neighbors that are within the separation range.
        # i.e. negative mean of weighted (1/distance) difference vectors to neighbors.
        separation = -np.sum(
            differences / distances**2 * (distances < self.separation_range),
            axis=1,
        )
        separation = normalize(separation)

        # 2. Alignment: Steer in the same direction as neighbors,
        # i.e. mean/sum of velocities of neighbors
        alignment = np.sum(self.velocities * (distances < self.alignment_range), axis=1)
        alignment = normalize(alignment)

        # 3. Cohesion: teer towards center of neighbors,
        # i.e. mean of difference vectors to neighbors
        cohesion = np.sum(
            differences / distances * (distances < self.cohesion_range), axis=1
        )
        cohesion = normalize(cohesion)

        # Compute desired velocities as weighted average.
        ws, wa, wc = self.steering_weights
        desired_velocities = (ws * separation + wa * alignment + wc * cohesion) / (
            ws + wa + wc
        )

        return desired_velocities

    def _compute_obstacle_bounce(self) -> np.ndarray:
        """Compute the obstacle avoidance velocities.

        Currently this only includes the world edge.

        These are 1 at the point of touching the obstacle and >1 when intersecting.
        When scaled to maximum velocity, this ensures the desired velocities are overpowered.
        """

        # Distances to 0 edge `xp` and width/height edge `xn`.
        xp = self.positions[:, [0]]
        yp = self.positions[:, [1]]
        xn = self.world_size[0] - xp
        yn = self.world_size[1] - yp

        bounce = np.zeros_like(self.velocities)
        # Positive x direction
        dxp = 1 - (xp - self.radius) / (self.obstacle_margin - self.radius)
        bounce += np.array([[1, 0]]) * (dxp > 0) * dxp
        # Negative x direction
        dxn = 1 - (xn - self.radius) / (self.obstacle_margin - self.radius)
        bounce += np.array([[-1, 0]]) * (dxn > 0) * dxn
        # Positive y direction
        dyp = 1 - (yp - self.radius) / (self.obstacle_margin - self.radius)
        bounce += np.array([[0, 1]]) * (dyp > 0) * dyp
        # Negative y direction
        dyn = 1 - (yn - self.radius) / (self.obstacle_margin - self.radius)
        bounce += np.array([[0, -1]]) * (dyn > 0) * dyn

        return limit(bounce, 1)
