# Hacky way to allow swarm import.
import sys

sys.path.append(".")

from pathlib import Path

import numpy as np

from experiments.random_to_target.env import create_curriculum
from swarm.bas import wrappers
from swarm.bas.wrappers.observation import components
from swarm.training.training import train

curriculum = create_curriculum(
    [
        components.SectionObservationComponent(num_sections=8, max_range=20),
        components.SectionObservationComponent(
            num_sections=8, max_range=20, offset_angle=np.pi / 8
        ),
        components.AgentVelocityObservationComponent(),
    ],
    wrappers.DesiredVelocityActionWrapper,
    {},
)


if __name__ == "__main__":
    train(
        curriculum,
        Path(__file__).relative_to(Path.cwd() / "experiments").parent
        / "section_angular",
        num_processes=16,
        video_every_n_steps=500000,
    )
