import argparse
import importlib
import sys
from pathlib import Path

import gymnasium as gym

sys.modules["gym"] = gym


from stable_baselines3 import PPO


def parse_model_env_window_scale() -> tuple[PPO, gym.Env, float]:
    parser = argparse.ArgumentParser(description="Run a trained model in an environment.")
    parser.add_argument("model_path", type=Path)
    parser.add_argument("env_path", type=Path, help="Path of the file in which the env_function is defined.")
    parser.add_argument("env_function", type=str, help="Name of the env_function.")
    parser.add_argument("env_call_with_args", type=str, help='env_function call (with arguments if necessary) in quotes, e.g. "()".')
    parser.add_argument("--window_scale", type=float, default=5)
    args = parser.parse_args()

    module = importlib.import_module(
        str(args.env_path).replace("/", ".").replace(".py", "")
    )
    env = eval(f"module.{args.env_function}{args.env_call_with_args}")

    model = PPO.load(args.model_path)

    return model, env, args.window_scale
