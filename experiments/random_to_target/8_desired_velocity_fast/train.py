# Hacky way to allow swarm import.
import sys
from typing import Type

sys.path.append(".")

from pathlib import Path

from experiments.random_to_target.env import create_env
from bas import wrappers
from bas.wrappers.observation import components
from training.training import train


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
                max_boid_velocity=0.5,
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
                max_boid_velocity=0.6,
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
                max_boid_velocity=0.7,
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
                max_boid_velocity=0.8,
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
                max_boid_velocity=0.9,
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
                following_weight=0.2,
                world_size=(200, 200),
                num_boids=100,
                max_boid_velocity=0.9,
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
                following_weight=0.2,
                world_size=(200, 200),
                num_boids=100,
                max_boid_velocity=0.9,
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
                following_weight=0.2,
                world_size=(200, 200),
                num_boids=100,
                max_boid_velocity=0.9,
                time_limit=500,
            ),
        ),
    ]


section_distance_curriculum = create_curriculum(
    [
        components.SectionDistanceObservationComponent(num_sections=8, max_range=20),
        components.AgentVelocityObservationComponent(),
    ],
    wrappers.DesiredVelocityActionWrapper,
    {},
)
section_distance_and_section_velocity_distance_curriculum = create_curriculum(
    [
        components.SectionDistanceObservationComponent(num_sections=8, max_range=20),
        components.SectionVelocityDistanceObservationComponent(
            num_sections=8, max_range=20
        ),
        components.AgentVelocityObservationComponent(),
    ],
    wrappers.DesiredVelocityActionWrapper,
    {},
)


if __name__ == "__main__":
    train(
        section_distance_curriculum,
        Path(__file__).relative_to(Path.cwd() / "experiments").parent
        / "section_distance",
        num_processes=8,
        video_every_n_steps=500000,
    )
    train(
        section_distance_and_section_velocity_distance_curriculum,
        Path(__file__).relative_to(Path.cwd() / "experiments").parent
        / "section_distance_and_section_velocity_distance",
        num_processes=8,
        video_every_n_steps=500000,
    )
