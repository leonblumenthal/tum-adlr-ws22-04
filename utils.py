import argparse
import importlib
import sys
from pathlib import Path

import gymnasium as gym

sys.modules["gym"] = gym


import numpy as np
from stable_baselines3 import PPO


def parse_model_env_window_scale() -> tuple[PPO, gym.Env, float]:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=Path)
    parser.add_argument("env_path", type=Path)
    parser.add_argument("env_thing", type=str)
    parser.add_argument("env_call", type=str)
    parser.add_argument("--window_scale", type=float, default=5)
    args = parser.parse_args()

    module = importlib.import_module(
        str(args.env_path).replace("/", ".").replace(".py", "")
    )
    env = eval(f"module.{args.env_thing}{args.env_call}")

    model = PPO.load(args.model_path)

    return model, env, args.window_scale
