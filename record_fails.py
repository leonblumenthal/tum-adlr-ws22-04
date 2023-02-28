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


def record(
    model: PPO, env: gym.Env, window_scale: float, video_path: Path, num_episodes: int
):
    inject_render_wrapper(env, window_scale=window_scale)
    observation, info = env.reset()

    saved_frames = []

    frames = []
    while num_episodes > 0:
        agent_action, _ = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(agent_action)

        frames.append(env.render()[:, :, ::-1])

        if terminated or truncated:
            if not info.get("is_success", False):
                saved_frames += frames
                num_episodes -= 1
                print(len(saved_frames))
            observation, info = env.reset()
            frames = []

    video_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(
        str(video_path), fourcc, 10, saved_frames[0].shape[:2]
    )

    for frame in saved_frames:
        video_writer.write(frame)
    video_writer.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=Path)
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

    model = PPO.load(args.model_path)

    record(model, env, args.window_scale, args.video_path, args.num_episodes)
