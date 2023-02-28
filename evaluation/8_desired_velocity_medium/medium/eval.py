import sys

sys.path.insert(0, ".")

from pathlib import Path

from evaluation.evaluate import evaluate
from experiments.random_to_target.env import create_env as create_training_env
from bas.wrappers.observation import components
from bas import wrappers


def create_env(observation_components):
    return create_training_env(
        observation_components + [components.AgentVelocityObservationComponent()],
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
        max_boid_velocity=0.6,
        time_limit=500,
    )


def section_distance():
    return create_env(
        [
            components.SectionDistanceObservationComponent(
                num_sections=8, max_range=20
            ),
        ],
    )


def section_distance_and_section_velocity_distance():
    return create_env(
        [
            components.SectionDistanceObservationComponent(
                num_sections=8, max_range=20
            ),
            components.SectionVelocityDistanceObservationComponent(
                num_sections=8, max_range=20
            ),
        ],
    )


if __name__ == "__main__":
    for name in [
        "section_distance",
        "section_distance_and_section_velocity_distance",
    ]:
        df = evaluate(
            f"runs/random_to_target/8_desired_velocity_medium/{name}/model_8.zip",
            eval(name),
            num_episodes=10000,
            num_processes=48,
        )
        df.to_csv(Path(__file__).parent / f"{name}.csv")
        print(name)
