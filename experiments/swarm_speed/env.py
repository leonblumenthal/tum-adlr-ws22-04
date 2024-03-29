import gymnasium as gym
import numpy as np

from bas import Agent, BASEnv, Blueprint, Swarm, wrappers
from bas.swarm import InstantSpawner, SwarmConfig
from bas.wrappers.observation import components


def create_env(
    target_radius=3,
    target_reward=3,
    distance_reward_transform=lambda d: -d,
    collision_termination=False,
    collision_reward=0,
    boid_max_velocity=0.6,
    boid_max_acceleration=0.2,
    num_sections=8,
    separation_range=10,
    cohesion_range=20,
    alignment_range=20,
    steering_weights=(4.0, 2.0, 1.0, 0.0),
    world_size=np.array([200, 200]),
    trajectories = True,
):
    blueprint = Blueprint(
        world_size=world_size,
    )
    agent = Agent(
        radius=1,
        max_velocity=1.0,
        max_acceleration=0.3,
    )
    swarm = Swarm(
        SwarmConfig(
            num_boids=100,
            radius=2,
            max_velocity=boid_max_velocity,
            max_acceleration=boid_max_acceleration,
            separation_range=separation_range,
            cohesion_range=cohesion_range,
            alignment_range=alignment_range,
            steering_weights=steering_weights,
            obstacle_margin=3,
            need_for_speed=0.2,
        ),
        InstantSpawner(),
    )
    env = BASEnv(blueprint, agent, swarm)

    env = wrappers.RandomTargetWrapper(env)
    target = env.target

    env = wrappers.TargetRewardWrapper(
        env,
        position=target,
        target_radius=target_radius,
        target_reward=target_reward,
        distance_reward_transform=distance_reward_transform,
    )
    env = wrappers.BoidCollisionWrapper(
        env,
        collision_termination=collision_termination,
        collision_reward=collision_reward,
        add_reward=True,
    )

    env = wrappers.ObservationContainerWrapper(
        env,
        [
            components.SectionDistanceObservationComponent(num_sections, 20),
            components.SectionVelocityDistanceObservationComponent(num_sections, 20),
            components.TargetDirectionDistanceObservationComponent(target,world_size[0]),
            components.AgentVelocityObservationComponent(),
        ],
    )

    env = gym.wrappers.TimeLimit(env, 1000)

    env = wrappers.FlattenObservationWrapper(env)

    env = wrappers.AngularAndVelocityActionWrapper(env)

    if trajectories:
        env = wrappers.TrajectoryWrapper(env)

    return env
