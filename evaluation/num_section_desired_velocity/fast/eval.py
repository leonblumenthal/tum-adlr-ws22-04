import sys

sys.path.insert(0, ".")

from pathlib import Path

from evaluation.evaluate import evaluate
from experiments.random_to_target.env import create_env as create_training_env
from swarm.bas.wrappers.observation import components
from swarm.bas import wrappers


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
        max_boid_velocity=0.9,
        time_limit=500,
    )


def section_3():
    return create_env(
        [
            components.SectionObservationComponent(num_sections=3, max_range=20),
        ],
    )


def section_5():
    return create_env(
        [
            components.SectionObservationComponent(num_sections=5, max_range=20),
        ],
    )


def section_12():
    return create_env(
        [
            components.SectionObservationComponent(num_sections=12, max_range=20),
        ],
    )


def section_16():
    return create_env(
        [
            components.SectionObservationComponent(num_sections=16, max_range=20),
        ],
    )


if __name__ == "__main__":
    for name in [
        "3",
        "5",
        "12",
        "16",
    ]:
        df = evaluate(
            f"runs/random_to_target/num_section_desired_velocity/{name}/model_8.zip",
            eval("section_" + name),
            num_episodes=10000,
            num_processes=48,
        )
        df.to_csv(Path(__file__).parent / f"{name}.csv")
        print(name)
