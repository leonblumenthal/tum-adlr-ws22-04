import argparse
import importlib
import sys

import gymnasium as gym

sys.modules["gym"] = gym
from pathlib import Path

import cv2
from stable_baselines3 import PPO
from tqdm import tqdm

from swarm.bas.render.utils import inject_render_wrapper


def record(
    model: PPO, env: gym.Env, window_scale: float, video_path: Path, num_episodes: int
):
    env = inject_render_wrapper(env, window_scale=window_scale)
    observation, info = env.reset()

    video_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(str(video_path), fourcc, 30, env.render().shape[:2])

    while num_episodes > 0:
        agent_action, _ = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(agent_action)

        video_writer.write(env.render()[:,:,::-1])

        if terminated or truncated:
            observation, info = env.reset()
            num_episodes -= 1

    video_writer.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=Path)
    parser.add_argument("env_path", type=Path, help="Path of the file in which the env_function is defined.")
    parser.add_argument("env_function", type=str, help="Name of the env_function.")
    parser.add_argument("env_call_with_args", type=str, help='env_function call (with arguments if necessary) in quotes, e.g. "()".')
    parser.add_argument("video_path", type=Path, help="Destination path where to store video.")
    parser.add_argument("num_episodes", type=int, help="Video length in episodes.",default=1)
    parser.add_argument("--window_scale", type=float, default=5)
    args = parser.parse_args()

    module = importlib.import_module(
        str(args.env_path).replace("/", ".").replace(".py", "")
    )
    env = eval(f"module.{args.env_function}{args.env_call_with_args}")

    model = PPO.load(args.model_path)

    record(model, env, args.window_scale, args.video_path, args.num_episodes)