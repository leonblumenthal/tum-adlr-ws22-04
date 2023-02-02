import gymnasium as gym
import numpy as np

from swarm.bas import Agent, BASEnv, Blueprint, Swarm, wrappers
from swarm.bas.swarm import InstantSpawner, SwarmConfig
from swarm.bas.wrappers.observation import components


def create_env():
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
            steering_weights=(1.1, 1, 1, 0.0),
            obstacle_margin=3,
        ),
        InstantSpawner(),
        reset_between_episodes=False,
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

    env = wrappers.FlattenObservationWrapper(env)

    env = wrappers.RelativeActionWrapper(env)

    env = wrappers.TrajectoryWrapper(env)

    return env
