"""Play the BAS environment with some wrappers, especially rendering."""

import argparse
import importlib
from pathlib import Path

import gymnasium as gym
from gymnasium.utils.play import play

from bas import wrappers
from bas.render.utils import inject_render_wrapper
from bas.wrappers import utils

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("env_path", type=Path, help="Path of the file in which the env_function is defined.")
    parser.add_argument("env_function", type=str, help="Name of the env_function.")
    parser.add_argument("env_call_with_args", type=str, help='env_function call (with arguments if necessary) in quotes, e.g. "()".')
    parser.add_argument("--window_scale", type=float, default=5)
    args = parser.parse_args()

    module = importlib.import_module(
        str(args.env_path).replace("/", ".").replace(".py", "")
    )
    env = eval(f"module.{args.env_function}{args.env_call_with_args}")
    env = inject_render_wrapper(env, window_scale=args.window_scale)

    env = utils.replace_wrapper(
        env,
        gym.ActionWrapper,
        wrappers.DesiredVelocityActionWrapper,
        dict(in_agent_frame=False),
    )

    env = wrappers.DiscreteActionWrapper(env, 5)
    play(
        env,
        keys_to_action=dict(
            z=0,  # no movement
            d=1,  # right
            w=2,  # up
            a=3,  # left
            s=4,  # down
        ),
    )
