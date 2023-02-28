# Hacky way to allow swarm import.
import sys

sys.path.append(".")

from pathlib import Path

import gymnasium as gym
import numpy as np

from bas import Agent, BASEnv, Blueprint, Swarm, wrappers
from bas.swarm import InstantSpawner, SwarmConfig
from bas.wrappers.action.angular import AngularActionWrapper
from bas.wrappers.observation import components
from training.training import train


def create_env():
    blueprint = Blueprint(
        world_size=np.array([200, 200]),
    )
    agent = Agent(
        radius=1,
        max_velocity=1,
        max_acceleration=0.2,
        reset_position=np.array([10.0, 100.0]),
    )
    swarm = Swarm(
        SwarmConfig(
            num_boids=1,
            radius=2,
            max_velocity=None,
            max_acceleration=0.1,
            separation_range=10,
            cohesion_range=20,
            alignment_range=20,
            steering_weights=(1.1, 0.5, 0.2, 0.0),
            obstacle_margin=3,
            need_for_speed=0.2,
        ),
        InstantSpawner(np.array([[500, 500]])),
    )
    env = BASEnv(blueprint, agent, swarm)

    target = np.array([190, 100.0])

    env = wrappers.TargetRewardWrapper(
        env,
        position=target,
        target_radius=3,
        target_reward=3,
        distance_reward_transform=lambda d: -d,
    )

    env = wrappers.ObservationContainerWrapper(
        env,
        [
            components.TargetDirectionObservationComponent(target),
        ],
    )

    env = gym.wrappers.TimeLimit(env, 1000)

    env = wrappers.FlattenObservationWrapper(env)

    env = AngularActionWrapper(env)

    env = wrappers.TrajectoryWrapper(env)

    return env


curriculum = [
    (500000, create_env),
    (500000, create_env),
    (500000, create_env),
    (500000, create_env),
]

if __name__ == "__main__":
    experiment_path = Path(__file__).relative_to(Path.cwd() / "experiments").parent
    train(curriculum, experiment_path, num_processes=6, video_every_n_steps=500000)
