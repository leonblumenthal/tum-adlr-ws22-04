import argparse
import os
import sys

import gymnasium as gym

sys.modules["gym"] = gym


# TODO: Specific install of stable_baselines3 compatible to gymnasium
from stable_baselines3 import PPO

from swarm.experiments import GoalInsideGridEnv


# TODO: WIP
def main(parameters_path: str):
    """Train PPO on GoalInsideGridEnv."""
    env = GoalInsideGridEnv()

    model = PPO.load(parameters_path)

    env = gym.wrappers.HumanRendering(env)

    obs, info = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, terminated, truncated, info = env.step(action)

        env.render()

        if terminated or truncated:
            obs, info = env.reset()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_name", type=str)
    parser.add_argument("--experiments_directory", type=str, default="experiments")

    args = parser.parse_args()

    parameters_path = os.path.join(args.experiments_directory, args.experiment_name)

    main(parameters_path)
