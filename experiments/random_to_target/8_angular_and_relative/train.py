# Hacky way to allow swarm import.
import sys

sys.path.append(".")

from pathlib import Path

from experiments.random_to_target.env import create_curriculum
from bas import wrappers
from bas.wrappers.observation import components
from training.training import train


section_angular_curriculum = create_curriculum(
    [
        components.SectionObservationComponent(num_sections=8, max_range=20),
        components.AgentVelocityObservationComponent(),
    ],
    wrappers.AngularActionWrapper,
    {},
)
section_angular_and_velocity_curriculum = create_curriculum(
    [
        components.SectionObservationComponent(num_sections=8, max_range=20),
        components.AgentVelocityObservationComponent(),
    ],
    wrappers.AngularAndVelocityActionWrapper,
    {},
)
section_and_velocity_relative_curriculum = create_curriculum(
    [
        components.SectionObservationComponent(num_sections=8, max_range=20),
        components.SectionVelocityObservationComponent(num_sections=8, max_range=20, relative_to_agent=True),
        components.AgentVelocityObservationComponent(),
    ],
    wrappers.DesiredVelocityActionWrapper,
    {},
)
section_distance_and_section_velocity_distance_relative_curriculum = create_curriculum(
    [
        components.SectionDistanceObservationComponent(num_sections=8, max_range=20),
        components.SectionVelocityDistanceObservationComponent(
            num_sections=8, max_range=20, relative_to_agent=True
        ),
        components.AgentVelocityObservationComponent(),
    ],
    wrappers.DesiredVelocityActionWrapper,
    {},
)


if __name__ == "__main__":
    train(
        section_angular_curriculum,
        Path(__file__).relative_to(Path.cwd() / "experiments").parent / "section_angular",
        num_processes=8,
        video_every_n_steps=500000,
    )
    train(
        section_angular_and_velocity_curriculum,
        Path(__file__).relative_to(Path.cwd() / "experiments").parent / "section_angular_and_velocity",
        num_processes=8,
        video_every_n_steps=500000,
    )
    train(
        section_and_velocity_relative_curriculum,
        Path(__file__).relative_to(Path.cwd() / "experiments").parent / "section_and_velocity_relative",
        num_processes=8,
        video_every_n_steps=500000,
    )
    train(
        section_distance_and_section_velocity_distance_relative_curriculum,
        Path(__file__).relative_to(Path.cwd() / "experiments").parent / "section_distance_and_section_velocity_distance_relative",
        num_processes=8,
        video_every_n_steps=500000,
    )
