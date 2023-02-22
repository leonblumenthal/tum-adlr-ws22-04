import argparse
import importlib
import sys
from pathlib import Path

import gymnasium as gym

sys.modules["gym"] = gym


from stable_baselines3 import PPO


def parse_model_env_window_scale() -> tuple[PPO, gym.Env, float]:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=Path)
    parser.add_argument("env_path", type=Path)
    parser.add_argument("env_function", type=str)
    parser.add_argument("env_call_with_args", type=str)
    parser.add_argument("--window_scale", type=float, default=5)
    args = parser.parse_args()

    module = importlib.import_module(
        str(args.env_path).replace("/", ".").replace(".py", "")
    )
    env = eval(f"module.{args.env_function}{args.env_call_with_args}")

    model = PPO.load(args.model_path)

    return model, env, args.window_scale
