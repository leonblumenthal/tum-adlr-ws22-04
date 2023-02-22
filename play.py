"""Play the BAS environment with some wrappers, especially rendering."""

import argparse
import importlib
from pathlib import Path

import gymnasium as gym
from gymnasium.utils.play import play

from swarm.bas import wrappers
from swarm.bas.render.utils import inject_render_wrapper
from swarm.bas.wrappers import utils

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("env_path", type=Path)
    parser.add_argument("env_thing", type=str)
    parser.add_argument("env_call", type=str)
    parser.add_argument("--window_scale", type=float, default=5)
    args = parser.parse_args()

    module = importlib.import_module(
        str(args.env_path).replace("/", ".").replace(".py", "")
    )
    env = eval(f"module.{args.env_thing}{args.env_call}")
    inject_render_wrapper(env, window_scale=args.window_scale)

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
