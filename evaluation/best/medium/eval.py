import sys

sys.path.insert(0, ".")

from pathlib import Path

import numpy as np

from evaluation.evaluate import evaluate
from experiments.random_to_target.env import create_env as create_training_env
from swarm.bas import wrappers
from swarm.bas.wrappers.observation import components


def create_env():
    return create_training_env(
        [
            components.SectionObservationComponent(num_sections=8, max_range=20),
            components.AgentVelocityObservationComponent(),
        ],
        action_wrapper_class=wrappers.AngularAndVelocityActionWrapper,
        action_wrapper_kwargs={},
        target_radius=3,
        target_reward=3,
        distance_reward_transform=lambda d: -(d**0.5) / 500,
        collision_termination=True,
        collision_reward=-3,
        following_weight=0.2,
        world_size=(200, 200),
        num_boids=100,
        max_boid_velocity=0.6,
        time_limit=500,
    )




if __name__ == "__main__":
    df = evaluate(
        f"runs/random_to_target/best/model_8.zip",
        create_env,
        num_episodes=10000,
        num_processes=48,
    )
    df.to_csv(Path(__file__).parent / f"best.csv")