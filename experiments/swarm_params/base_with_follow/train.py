# Hacky way to allow swarm import.
import sys

sys.path.append(".")

from pathlib import Path

from experiments.swarm_params.env import create_env
from swarm.training.training import train

curriculum = [
    (
        1000000,
        lambda: create_env(
            num_sections=8,
            distance_reward_transform=lambda d: -d,
            collision_termination=False,
            collision_reward=0,
            steering_weights=(4.0, 2.0, 1.0, 1.0),
        ),
    ),
    (
        1000000,
        lambda: create_env(
            num_sections=8,
            distance_reward_transform=lambda d: -d,
            collision_termination=False,
            collision_reward=-0.2,
            steering_weights=(4.0, 2.0, 1.0, 1.0),
        ),
    ),
    (
        1000000,
        lambda: create_env(
            num_sections=8,
            distance_reward_transform=lambda d: -d,
            collision_termination=False,
            collision_reward=-0.5,
            steering_weights=(4.0, 2.0, 1.0, 1.0),
        ),
    ),
    (
        1000000,
        lambda: create_env(
            num_sections=8,
            distance_reward_transform=lambda d: -d,
            collision_termination=False,
            collision_reward=-1,
            steering_weights=(4.0, 2.0, 1.0, 1.0),
        ),
    ),
    (
        1000000,
        lambda: create_env(
            num_sections=8,
            distance_reward_transform=lambda d: -d,
            collision_termination=False,
            collision_reward=-2,
            steering_weights=(4.0, 2.0, 1.0, 1.0),
        ),
    ),
    (
        500000,
        lambda: create_env(
            num_sections=8,
            distance_reward_transform=lambda d: -d,
            collision_termination=False,
            collision_reward=-3,
            steering_weights=(4.0, 2.0, 1.0, 1.0),
        ),
    ),
    (
        1000000,
        lambda: create_env(
            num_sections=8,
            distance_reward_transform=lambda d: -d / 1000 * 3,
            collision_termination=True,
            collision_reward=-3,
            steering_weights=(4.0, 2.0, 1.0, 1.0),
        ),
    ),
    (
        1000000,
        lambda: create_env(
            num_sections=8,
            distance_reward_transform=lambda d: -d / 1000 * 3,
            collision_termination=True,
            collision_reward=-3,
            steering_weights=(4.0, 2.0, 1.0, 1.0),
        ),
    ),
    (
        1000000,
        lambda: create_env(
            num_sections=8,
            distance_reward_transform=lambda d: -d / 1000 * 3,
            collision_termination=True,
            collision_reward=-3,
            steering_weights=(4.0, 2.0, 1.0, 1.0),
        ),
    ),
]

if __name__ == "__main__":
    experiment_path = Path(__file__).relative_to(Path.cwd() / "experiments").parent
    train(curriculum, experiment_path, num_processes=16, video_every_n_steps=1000000)