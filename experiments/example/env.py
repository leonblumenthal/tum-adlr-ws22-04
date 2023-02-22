import gymnasium as gym
import numpy as np

from swarm.bas import Agent, BASEnv, Blueprint, Swarm, wrappers
from swarm.bas.swarm import InstantSpawner, SwarmConfig
from swarm.bas.wrappers.observation import components


def create_env(
        num_sections=8,
        distance_reward_transform=lambda d: -d,
        collision_termination=False,
        collision_reward=0,
        trajectory=False,
):
    world_size = np.array([200, 200])

    blueprint = Blueprint(
        world_size=world_size,
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
            steering_weights=(1.1, 1, 1, 0.0),
            obstacle_margin=3,
        ),
        InstantSpawner(),
        reset_between_episodes=True,
    )
    env = BASEnv(blueprint, agent, swarm)

    env = wrappers.RandomTargetWrapper(env)
    target = env.target

    env = wrappers.TargetRewardWrapper(
        env,
        position=target,
        target_radius=3,
        target_reward=3,
        distance_reward_transform=distance_reward_transform,
    )
    env = wrappers.BoidCollisionWrapper(
        env,
        collision_termination=collision_termination,
        collision_reward=collision_reward,
    )

    env = wrappers.ObservationContainerWrapper(
        env,
        [
            components.SectionDistanceObservationComponent(num_sections, 20),
            components.SectionVelocityDistanceObservationComponent(num_sections, 20),
            components.TargetDirectionDistanceObservationComponent(target, world_size[0]),
            components.AgentVelocityObservationComponent(),
        ],
    )

    env = gym.wrappers.TimeLimit(env, 1000)
    
    env = wrappers.FlattenObservationWrapper(env)

    env = wrappers.AngularAndVelocityActionWrapper(env)

    if trajectory:
        env = wrappers.TrajectoryWrapper(env)


    return env
