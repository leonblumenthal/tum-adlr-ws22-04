from dataclasses import dataclass

import numpy as np


@dataclass
class SwarmConfig:
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
        field_of_view: Field of view of a boid centered around its velocity direction.
    """

    num_boids: int
    radius: float
    max_velocity: float | None
    max_acceleration: float
    separation_range: float
    alignment_range: float
    cohesion_range: float
    steering_weights: tuple[float, float, float, float]
    obstacle_margin: float
    target_position: np.ndarray | None = None
    target_radius: float = 10
    target_range: float | None = None
    target_despawn: bool = False
    field_of_view: float = 2 * np.pi
