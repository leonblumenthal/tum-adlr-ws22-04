"""Script for playing an environment with direction vectors as actions."""

import argparse

import gymnasium as gym
from gymnasium.envs.registration import register
from gymnasium.utils.play import play
import numpy as np


def main(rel_env_path: str, env_class_name: str, seed: int = None):
    entry_point = f"{rel_env_path[:-3].replace('/', '.')}:{env_class_name}"

    register(env_class_name, entry_point)
    env = gym.make(env_class_name, render_mode="rgb_array")

    play(
        env,
        keys_to_action=dict(
            z=np.array([0, 0]),  # no movement
            d=np.array([1, 0]),  # right
            w=np.array([0, 1]),  # up
            a=np.array([-1, 0]),  # left
            s=np.array([0, -1]),  # down
        ),
        seed=seed,
        # Print reward.
        callback=lambda *x: print(x[3], x[-1]),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("rel_env_path", help="Relative path of the environment")
    parser.add_argument("env_class_name", help="Class name of the environment")
    parser.add_argument("--seed")

    args = parser.parse_args()

    main(args.rel_env_path, args.env_class_name, args.seed)
