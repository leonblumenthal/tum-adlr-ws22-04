import argparse
import pathlib
import sys

import gymnasium as gym

sys.modules["gym"] = gym
from stable_baselines3 import PPO

from swarm.experiments import RelativeDynamicRandomTargetEnv


def run_ppo(parameters_path: str):
    env = RelativeDynamicRandomTargetEnv()

    model = PPO.load(parameters_path)

    env = gym.wrappers.HumanRendering(env)

    obs, info = env.reset()
    while True:
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)

        env.render()

        if terminated or truncated:
            obs, info = env.reset()

        print(f"{reward=}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=pathlib.Path)
    args = parser.parse_args()

    run_ppo(args.model_path)
