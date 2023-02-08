from typing import Callable, Type
import gymnasium as gym
import numpy as np

from swarm.bas import Agent, BASEnv, Blueprint, Swarm, wrappers
from swarm.bas.swarm import InstantSpawner, SwarmConfig
from swarm.bas.wrappers.observation import components


def create_env(
    observation_components: list[components.ObservationComponent],
    action_wrapper_class: Type,
    action_wrapper_kwargs: dict,
    target_radius: float,
    target_reward: float,
    distance_reward_transform: Callable[[float], float],
    collision_termination: bool,
    collision_reward: float,
    following_weight: float,
    world_size: tuple[int, int],
    num_boids: int,
    max_boid_velocity: float | None,
    time_limit: int,
):
    blueprint = Blueprint(
        world_size=np.array(world_size),
    )
    agent = Agent(
        radius=1,
        max_velocity=1,
        max_acceleration=0.2,
    )
    swarm = Swarm(
        SwarmConfig(
            num_boids=num_boids,
            radius=2,
            max_velocity=max_boid_velocity,
            max_acceleration=0.1,
            separation_range=10,
            cohesion_range=20,
            alignment_range=20,
            steering_weights=(1.6, 0.5, 0.2, following_weight),
            obstacle_margin=3,
            need_for_speed=0.2,
            target_position=agent.position,
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
        observation_components
        + [
            components.TargetDirectionDistanceObservationComponent(
                target, np.linalg.norm(world_size), 0.1
            )
        ],
    )

    env = gym.wrappers.TimeLimit(env, time_limit)

    env = wrappers.FlattenObservationWrapper(env)

    env = action_wrapper_class(env, **action_wrapper_kwargs)

    env = wrappers.SpawnFixWrapper(env)

    env = wrappers.TrajectoryWrapper(env)

    return env


def create_curriculum(
    observation_components, action_wrapper_class: Type, action_wrapper_kwargs: dict
):
    return [
        # Only learn how to handle the default observation, i.e. go straight to goal.
        (
            250000,
            lambda: create_env(
                observation_components,
                action_wrapper_class=action_wrapper_class,
                action_wrapper_kwargs=action_wrapper_kwargs,
                target_radius=3,
                target_reward=3,
                distance_reward_transform=lambda d: -(d**0.5),
                collision_termination=False,
                collision_reward=0,
                following_weight=0,
                world_size=(100, 100),
                num_boids=0,
                max_boid_velocity=None,
                time_limit=500,
            ),
        ),
        (
            750000,
            lambda: create_env(
                observation_components,
                action_wrapper_class=action_wrapper_class,
                action_wrapper_kwargs=action_wrapper_kwargs,
                target_radius=3,
                target_reward=3,
                distance_reward_transform=lambda d: -(d**0.5),
                collision_termination=False,
                collision_reward=-1,
                following_weight=0,
                world_size=(100, 100),
                num_boids=40,
                max_boid_velocity=0.3,
                time_limit=500,
            ),
        ),
        (
            1000000,
            lambda: create_env(
                observation_components,
                action_wrapper_class=action_wrapper_class,
                action_wrapper_kwargs=action_wrapper_kwargs,
                target_radius=3,
                target_reward=3,
                distance_reward_transform=lambda d: -(d**0.5),
                collision_termination=False,
                collision_reward=-2,
                following_weight=0,
                world_size=(100, 100),
                num_boids=40,
                max_boid_velocity=0.3,
                time_limit=500,
            ),
        ),
        (
            1000000,
            lambda: create_env(
                observation_components,
                action_wrapper_class=action_wrapper_class,
                action_wrapper_kwargs=action_wrapper_kwargs,
                target_radius=3,
                target_reward=3,
                distance_reward_transform=lambda d: -(d**0.5),
                collision_termination=False,
                collision_reward=-3,
                following_weight=0,
                world_size=(100, 100),
                num_boids=40,
                max_boid_velocity=0.3,
                time_limit=500,
            ),
        ),
        (
            1000000,
            lambda: create_env(
                observation_components,
                action_wrapper_class=action_wrapper_class,
                action_wrapper_kwargs=action_wrapper_kwargs,
                target_radius=3,
                target_reward=3,
                distance_reward_transform=lambda d: -(d**0.5) / 500,
                collision_termination=True,
                collision_reward=-3,
                following_weight=0,
                world_size=(100, 100),
                num_boids=40,
                max_boid_velocity=0.3,
                time_limit=500,
            ),
        ),
        (
            1000000,
            lambda: create_env(
                observation_components,
                action_wrapper_class=action_wrapper_class,
                action_wrapper_kwargs=action_wrapper_kwargs,
                target_radius=3,
                target_reward=3,
                distance_reward_transform=lambda d: -(d**0.5) / 500,
                collision_termination=True,
                collision_reward=-3,
                following_weight=0.1,
                world_size=(200, 200),
                num_boids=100,
                max_boid_velocity=0.3,
                time_limit=500,
            ),
        ),
        (
            1000000,
            lambda: create_env(
                observation_components,
                action_wrapper_class=action_wrapper_class,
                action_wrapper_kwargs=action_wrapper_kwargs,
                target_radius=3,
                target_reward=3,
                distance_reward_transform=lambda d: -(d**0.5) / 500,
                collision_termination=True,
                collision_reward=-3,
                following_weight=0.1,
                world_size=(200, 200),
                num_boids=100,
                max_boid_velocity=0.3,
                time_limit=500,
            ),
        ),
        (
            1000000,
            lambda: create_env(
                observation_components,
                action_wrapper_class=action_wrapper_class,
                action_wrapper_kwargs=action_wrapper_kwargs,
                target_radius=3,
                target_reward=3,
                distance_reward_transform=lambda d: -(d**0.5) / 500,
                collision_termination=True,
                collision_reward=-3,
                following_weight=0.1,
                world_size=(200, 200),
                num_boids=100,
                max_boid_velocity=0.3,
                time_limit=500,
            ),
        ),
        (
            1000000,
            lambda: create_env(
                observation_components,
                action_wrapper_class=action_wrapper_class,
                action_wrapper_kwargs=action_wrapper_kwargs,
                target_radius=3,
                target_reward=3,
                distance_reward_transform=lambda d: -(d**0.5) / 500,
                collision_termination=True,
                collision_reward=-3,
                following_weight=0.1,
                world_size=(200, 200),
                num_boids=100,
                max_boid_velocity=0.3,
                time_limit=500,
            ),
        ),
    ]
