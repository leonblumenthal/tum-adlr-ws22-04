import argparse
import pathlib
import sys

import cv2
import gymnasium
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from tqdm import tqdm

sys.modules["gym"] = gymnasium
from stable_baselines3 import PPO

from swarm.bas import BASEnv
from swarm.experiments import GoalInsideGridEnv


def _get_trajectories(env: BASEnv, model: PPO, num_steps: int) -> list[np.ndarray]:
    obs, _ = env.reset()

    trajectories = [[]]
    for _ in tqdm(range(num_steps)):
        action, _ = model.predict(obs)
        obs, _, terminated, truncated, _ = env.step(action)

        trajectories[-1].append(env.agent.position)
        if terminated or truncated:
            trajectories.append([])

        if terminated or truncated:
            obs, _ = env.reset()

    return [np.array(positions) for positions in trajectories]


def _draw_trajectories(
    image: np.ndarray, trajectories: list[np.ndarray], stroke_width: int
):
    for positions in trajectories:
        for a, b in zip(positions, positions[1:]):
            cv2.line(
                image,
                a.astype(int),
                b.astype(int),
                (0, 0, 0),
                stroke_width,
            )


def create_trajectories_plot(
    model_path: str,
    num_steps: int,
    window_scale: float,
    stroke_width: int,
    save_path: pathlib.Path | None,
):
    """Plot or save trajectories for static environment."""

    # TODO: Make configurable
    env = GoalInsideGridEnv(window_scale=window_scale)

    model = PPO.load(model_path)

    trajectories = _get_trajectories(env, model, num_steps)
    trajectories = [positions * env.window_scale for positions in trajectories]

    # Move agent off screen for rendering.
    env.agent.position = env.blueprint.world_size * 2
    image = env.render()[::-1].copy()

    _draw_trajectories(image, trajectories, stroke_width)

    if save_path is None:
        fig, ax = plt.subplots(dpi=200)
        ax.imshow(image[::-1])
        ax.axis("off")
        ax.margins(0, 0)
        fig.tight_layout()
        fig.show()
        plt.show()
    else:
        plt.imsave(save_path, image[::-1])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str)
    parser.add_argument("--num_steps", type=int, default=10000)
    parser.add_argument("--window_scale", type=float, default=5)
    parser.add_argument("--stroke_width", type=int, default=1)
    parser.add_argument("--save_path", type=pathlib.Path)
    args = parser.parse_args()

    create_trajectories_plot(
        args.model_path,
        args.num_steps,
        args.window_scale,
        args.stroke_width,
        args.save_path,
    )
