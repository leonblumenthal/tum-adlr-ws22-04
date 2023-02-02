"""Play the BAS environment with some wrappers, especially rendering."""

from gymnasium.utils.play import play
import numpy as np
import gymnasium as gym

from swarm.bas import wrappers
from swarm.bas import Agent, BASEnv, Blueprint, RenderWrapper, Swarm, wrappers
from swarm.bas.swarm import InstantSpawner, SwarmConfig
from swarm.bas.wrappers.observation import components

blueprint = Blueprint(
    world_size=np.array([200, 200]),
)
agent = Agent(
    radius=1,
    max_velocity=1,
    max_acceleration=0.2,
)
swarm = Swarm(
    SwarmConfig(
        num_boids=100,
        radius=2,
        max_velocity=1,
        max_acceleration=0.1,
        separation_range=10,
        cohesion_range=20,
        alignment_range=20,
        steering_weights=(3.1, 2, 1, 0),
        obstacle_margin=3,
        # target_position=agent.position,
        # field_of_view=np.pi * 1.5,
        need_for_speed=0.5
    ),
    InstantSpawner(),
)
env = BASEnv(blueprint, agent, swarm)


env = wrappers.RandomTargetWrapper(env)
target = env.target

env = wrappers.TargetRewardWrapper(
    env,
    position=target,
    target_radius=3,
    target_reward=3,
    distance_reward_transform=lambda d: -d,
)
env = wrappers.BoidCollisionWrapper(
    env,
    collision_termination=False,
    collision_reward=0,
    add_reward=True,
)

env = wrappers.ObservationContainerWrapper(
    env,
    [
        components.SectionDistanceObservationComponent(8, 20),
        components.SectionVelocityDistanceObservationComponent(
            8, 20, relative_to_agent=False
        ),
        components.TargetDirectionObservationComponent(target),
        components.AgentVelocityObservationComponent(),
    ],
)

env = gym.wrappers.TimeLimit(env, 1000)

env = RenderWrapper(env, window_scale=5)

env = wrappers.FlattenObservationWrapper(env)

# env = wrappers.RelativeActionWrapper(env)
env = wrappers.FlattenObservationWrapper(env)


env = wrappers.TrajectoryWrapper(env)

env = wrappers.DiscreteActionWrapper(env, 5)
play(
    env,
    keys_to_action=dict(
        z=0,  # no movement
        d=1,  # right
        w=2,  # up
        a=3,  # left
        s=4,  # down
    ),
)
