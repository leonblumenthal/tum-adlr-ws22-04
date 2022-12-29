import argparse
import os
import sys

import gymnasium

sys.modules["gym"] = gymnasium


# TODO: Specific install of stable_baselines3 compatible to gymnasium
from stable_baselines3 import PPO

from swarm.experiments import GoalInsideGridEnv


# TODO: WIP
def main(parameters_path: str, num_total_steps: int, load_existing: bool):
    """Train PPO on GoalInsideGridEnv."""
    env = GoalInsideGridEnv()

    model = PPO("MlpPolicy", env, verbose=1)
    if load_existing:
        model.set_parameters(parameters_path)

    model.learn(total_timesteps=num_total_steps)

    model.save(parameters_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_name", type=str)
    parser.add_argument("num_total_steps", type=int)
    parser.add_argument("--experiments_directory", type=str, default="experiments")
    parser.add_argument("--load_existing", action="store_true")

    args = parser.parse_args()

    os.makedirs(args.experiments_directory, exist_ok=True)
    parameters_path = os.path.join(args.experiments_directory, args.experiment_name)

    main(parameters_path, args.num_total_steps, args.load_existing)
