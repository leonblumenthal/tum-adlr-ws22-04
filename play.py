"""Script for playing an environment with a discrete action space"""

import argparse

import gymnasium as gym
from gymnasium.envs.registration import register
from gymnasium.utils.play import play


def main(
    rel_env_path: str,
    env_class_name: str,
    seed: int | None,
    keys_to_action: str,
):
    entry_point = (
        f"{rel_env_path[:-3].replace('/', '.')}:{env_class_name}"
    )

    register(env_class_name, entry_point)
    env = gym.make(env_class_name, render_mode="rgb_array")

    play(env, keys_to_action={c: i for i, c in enumerate(keys_to_action)}, seed=seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("rel_env_path", help="Relative path of the environment")
    parser.add_argument("env_class_name", help="Class name of the environment")
    parser.add_argument("--seed")
    parser.add_argument(
        "--kta",
        "--keys_to_action",
        help="Mapping of characters to discrete actions by index, e.g. abc -> a:0, b:1, c:2",
        default="zdwas",
    )

    args = parser.parse_args()

    main(args.rel_env_path, args.env_class_name, args.seed, args.kta)
