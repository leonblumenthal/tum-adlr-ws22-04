import numpy as np

from swarm import utils


class Swarm2D:
    """2D implementation of a swarm of boids.

    The methods perform similar to Gymnasium environments,
    i.e. the `reset` method initializes the boids and the `step` method updates them.

    Important:
    All internal operations, especially the distance computations,
    are implemented as dense matrix operations.
    This may lead to unecessary computational overhead if each boid's neighborhood
    only represents a small fraction of the entire swarm.
    """

    def __init__(
        self,
        num_boids: int,
        boid_radius: float,
        boid_max_speed: float,
        boid_max_acceleration: float,
        neighbor_range: float,
        seperation_range_fraction: float,
        steering_weights: tuple[float, float, float],
        world_size: np.ndarray,
        obstacle_margin: float,
        np_random: np.random.Generator | None = None,
    ):
        """Constructor of the `Swarm2D` class.

        Args:
            num_boids: Number of circular boids in the swarm.
            boid_radius: Radius of each boid in world size.
            boid_max_speed: Maximum speed of each boid.
            boid_max_acceleration: Maximum acceleration of each boid.
            neighbor_range: Maximum distance between boids to be considered neighbors.
            seperation_range_fraction: Fraction of `neighbor_range`, which separation rule considers.
            steering_weights: Weights for the three rules.
            world_size: Width and height of the world. [0, world_size[0]] x [0, world_size[1]]
            obstacle_margin: Distance to obstacles in which the boids are forced away.
            np_random: Numpy RNG, preferably from a Gymnaiusm environment. Defaults to None.
        """

        assert obstacle_margin > boid_radius, "should always hold"

        self.num_boids = num_boids
        self.boid_radius = boid_radius
        self.boid_max_speed = boid_max_speed
        self.boid_max_acceleration = boid_max_acceleration
        self.neighbor_range = neighbor_range
        self.separation_range_fraction = seperation_range_fraction
        self.steering_weights = steering_weights
        # World is [0, world_size[0]] x [0, world_size[1]].
        self.world_size = world_size
        self.obstacle_margin = obstacle_margin
        self.np_random = np.random if np_random is None else np_random

    def reset(self):
        """Reset boid positions and velocities to random initial state."""

        # Sample boid posisition randomly in world size.
        self.boid_positions = self.np_random.uniform(
            low=self.boid_radius,
            high=self.world_size - self.obstacle_margin,
            size=(self.num_boids, 2),
        )
        # Velocities from last step are needed for computation.
        self.boid_velocities = np.zeros_like(self.boid_positions)

    def step(self):
        """Update the position of each boid.

        Each boid individually follows 3 rules:
        1. Collision avoidance (separation)
        2. Velocity matching (alignment)
        3. Flock centering (cohesion)

        Additionally and obstacle/world edge avoidance is incorporated.

        The position is updated using the Euler method, i.e. v = v + a -> p = p + v.
        """

        # Compute desired velocities and resulting accelerations.
        desired_velocities = self._compute_desired_velocities() * self.boid_max_speed
        accelerations = desired_velocities - self.boid_velocities
        accelerations = utils.limit(accelerations, self.boid_max_acceleration)

        # Update velocities.
        self.boid_velocities += accelerations
        self.boid_velocities = utils.limit(self.boid_velocities, self.boid_max_speed)

        # Add velocities to avoid obstacles.
        self.boid_velocities += self._compute_obstacle_bounce() * self.boid_max_speed

        self.boid_positions += self.boid_velocities

    def _compute_desired_velocities(self) -> np.ndarray:
        """Compute the normalized desired velocities according to the three main rules."""

        # Compute pairwise difference vectors and distances.
        # difference[i, j] <=> vector from i to j.
        differences = self.boid_positions[None, :] - self.boid_positions[:, None, :]
        distances = np.linalg.norm(differences, axis=-1, keepdims=True)
        # Ensure that the distance to itself is greater than the neighbor threshold.
        distances += np.eye(distances.shape[0])[..., None] * self.neighbor_range

        # [i, j] == True <=> "i is in neighbor distance of j are neighbors" (and vice versa).
        neighbors = distances < self.neighbor_range

        # 1. Separation: Steer away from neighbors that are within the separation range.
        # i.e. negative mean of weighted (1/distance) difference vectors to neighbors.
        separation_neighbors = (
            distances < self.neighbor_range * self.separation_range_fraction
        )
        separation = -np.sum(
            differences / distances**2 * separation_neighbors,
            axis=1,
        )
        separation = utils.normalize(separation)

        # 2. Alignment: Steer in the same direction as neighbors,
        # i.e. mean/sum of velocities of neighbors
        alignment = np.sum(self.boid_velocities * neighbors, axis=1)
        alignment = utils.normalize(alignment)

        # 3. Cohesion: teer towards center of neighbors,
        # i.e. mean of difference vectors to neighbors
        cohesion = np.sum(differences / distances * neighbors, axis=1)
        cohesion = utils.normalize(cohesion)

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
        When scales to boid max speed, this ensures the desired velocities are overpowered.
        """

        # Distances to 0 edge `xp` and width/height edge `xn`.
        xp = self.boid_positions[:, [0]]
        yp = self.boid_positions[:, [1]]
        xn = self.world_size[0] - xp
        yn = self.world_size[1] - yp

        bounce = np.zeros_like(self.boid_velocities)
        # Positive x direction
        dxp = 1 - (xp - self.boid_radius) / (self.obstacle_margin - self.boid_radius)
        bounce += np.array([[1, 0]]) * (dxp > 0) * dxp
        # Negative x direction
        dxn = 1 - (xn - self.boid_radius) / (self.obstacle_margin - self.boid_radius)
        bounce += np.array([[-1, 0]]) * (dxn > 0) * dxn
        # Positive y direction
        dyp = 1 - (yp - self.boid_radius) / (self.obstacle_margin - self.boid_radius)
        bounce += np.array([[0, 1]]) * (dyp > 0) * dyp
        # Negative y direction
        dyn = 1 - (yn - self.boid_radius) / (self.obstacle_margin - self.boid_radius)
        bounce += np.array([[0, -1]]) * (dyn > 0) * dyn

        return utils.limit(bounce, 1)
