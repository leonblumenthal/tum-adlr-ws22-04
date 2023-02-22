"""Wrappers for BAS environments."""

from .boid_collision_wrapper import BoidCollisionWrapper
from .discrete_action import DiscreteActionWrapper
from .target_reward import TargetRewardWrapper
from .num_neighbors_reward import NumNeighborsRewardWrapper
from .trajectory_wrapper import TrajectoryWrapper
from .random_target_wrapper import RandomTargetWrapper
from .observation.container_wrapper import ObservationContainerWrapper
from .observation.flatten_wrapper import FlattenObservationWrapper
from .action.angular import AngularActionWrapper
from .action.angular_and_velocity import AngularAndVelocityActionWrapper
from .action.desired_velocity import DesiredVelocityActionWrapper
from .spawn_fix import SpawnFixWrapper