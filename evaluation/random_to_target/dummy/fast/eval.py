import sys

sys.path.insert(0, ".")

from pathlib import Path

import numpy as np

from evaluation.random_to_target.evaluate import evaluate
from experiments.random_to_target.env import create_env as create_training_env
from bas import wrappers
from bas.wrappers.observation import components


def create_env():
    return create_training_env(
        [],
        action_wrapper_class=wrappers.DesiredVelocityActionWrapper,
        action_wrapper_kwargs={},
        target_radius=3,
        target_reward=3,
        distance_reward_transform=lambda d: -(d**0.5) / 500,
        collision_termination=True,
        collision_reward=-3,
        following_weight=0.2,
        world_size=(200, 200),
        num_boids=100,
        max_boid_velocity=0.9,
        time_limit=500,
    )


if __name__ == "__main__":
    df = evaluate(
        f"dummy",
        create_env,
        num_episodes=10000,
        num_processes=48,
    )
    df.to_csv(Path(__file__).parent / f"dummy.csv")
