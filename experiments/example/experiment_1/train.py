# Hacky way to allow swarm import.
import sys

sys.path.append(".")

from pathlib import Path

from experiments.example.env import create_env
from swarm.training.training import train

curriculum = [(50000, create_env)]

if __name__ == "__main__":

    experiment_path = Path(__file__).relative_to(Path.cwd() / "experiments").parent
    train(curriculum, experiment_path, num_processes=6)
