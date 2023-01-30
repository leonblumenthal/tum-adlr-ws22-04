import sys

import cv2
import gymnasium
import numpy as np
from tqdm import tqdm

sys.modules["gym"] = gymnasium
from stable_baselines3 import PPO

from swarm.bas import BASEnv


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


def draw_trajectories(
    image: np.ndarray,
    trajectories: list[np.ndarray],
    stroke_width: int,
    stroke_color: tuple[int, int, int] = (0, 0, 0),
):
    for positions in trajectories:
        for position_1, positions_2 in zip(positions, positions[1:]):
            cv2.line(
                image,
                position_1.astype(int),
                positions_2.astype(int),
                stroke_color,
                stroke_width,
            )


def create_trajectories_image(
    model: str | PPO,
    env: BASEnv,
    num_steps: int = 100000,
    stroke_width: int = 1,
) -> np.ndarray:
    """Plot trajectories for a static environment."""

    if isinstance(model, str):
        model = PPO.load(model)

    trajectories = _get_trajectories(env, model, num_steps)
    trajectories = [positions * env.window_scale for positions in trajectories]

    # Draw background image of the environment with agent off screen.
    env.agent.position = env.blueprint.world_size * 2
    image = env.render()

    draw_trajectories(image, trajectories, stroke_width)

    return image
