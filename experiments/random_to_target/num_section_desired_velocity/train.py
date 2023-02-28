# Hacky way to allow swarm import.
import sys

sys.path.append(".")

from pathlib import Path

from experiments.random_to_target.env import create_curriculum
from bas import wrappers
from bas.wrappers.observation import components
from training.training import train


section_3_curriculum = create_curriculum(
    [
        components.SectionObservationComponent(num_sections=3, max_range=20),
        components.AgentVelocityObservationComponent(),
    ],
    wrappers.DesiredVelocityActionWrapper,
    {},
)
section_5_curriculum = create_curriculum(
    [
        components.SectionObservationComponent(num_sections=5, max_range=20),
        components.AgentVelocityObservationComponent(),
    ],
    wrappers.DesiredVelocityActionWrapper,
    {},
)
section_12_curriculum = create_curriculum(
    [
        components.SectionObservationComponent(num_sections=12, max_range=20),
        components.AgentVelocityObservationComponent(),
    ],
    wrappers.DesiredVelocityActionWrapper,
    {},
)
section_16_curriculum = create_curriculum(
    [
        components.SectionObservationComponent(num_sections=16, max_range=20),
        components.AgentVelocityObservationComponent(),
    ],
    wrappers.DesiredVelocityActionWrapper,
    {},
)


if __name__ == "__main__":
    train(
        section_3_curriculum,
        Path(__file__).relative_to(Path.cwd() / "experiments").parent / "3",
        num_processes=8,
        video_every_n_steps=500000,
    )
    train(
        section_5_curriculum,
        Path(__file__).relative_to(Path.cwd() / "experiments").parent / "5",
        num_processes=8,
        video_every_n_steps=500000,
    )
    train(
        section_12_curriculum,
        Path(__file__).relative_to(Path.cwd() / "experiments").parent / "12",
        num_processes=8,
        video_every_n_steps=500000,
    )
    train(
        section_16_curriculum,
        Path(__file__).relative_to(Path.cwd() / "experiments").parent / "16",
        num_processes=8,
        video_every_n_steps=500000,
    )
