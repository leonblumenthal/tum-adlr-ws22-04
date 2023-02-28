import argparse
import importlib
import sys

import gymnasium as gym

sys.modules["gym"] = gym
from pathlib import Path

import cv2
from stable_baselines3 import PPO
from tqdm import tqdm

from bas.render.utils import inject_render_wrapper


def record(env: gym.Env, window_scale: float, video_path: Path, num_episodes: int):
    inject_render_wrapper(env, window_scale=window_scale)
    observation, info = env.reset()

    env.agent.position += 1000
    env.agent.step = lambda *a: None

    video_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(str(video_path), fourcc, 30, env.render().shape[:2])

    while num_episodes > 0:
        observation, reward, terminated, truncated, info = env.step(
            env.action_space.sample()
        )

        video_writer.write(env.render()[:, :, ::-1])

        if terminated or truncated:
            observation, info = env.reset()
            num_episodes -= 1

    video_writer.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("env_path", type=Path)
    parser.add_argument("env_thing", type=str)
    parser.add_argument("env_call", type=str)
    parser.add_argument("video_path", type=Path)
    parser.add_argument("num_episodes", type=int)
    parser.add_argument("--window_scale", type=float, default=5)
    args = parser.parse_args()

    module = importlib.import_module(
        str(args.env_path).replace("/", ".").replace(".py", "")
    )
    env = eval(f"module.{args.env_thing}{args.env_call}")

    record(env, args.window_scale, args.video_path, args.num_episodes)
