import argparse
import pathlib
import sys

import gymnasium
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

sys.modules["gym"] = gymnasium
from stable_baselines3 import PPO

from swarm.bas import BASEnv
from swarm.experiments import GoalInsideGridEnv


def _get_positions(env: BASEnv, num_width: int, num_height: int) -> np.ndarray:
    width, height = env.blueprint.world_size
    positions = np.stack(
        np.meshgrid(
            (np.arange(num_width) + 0.5) / num_width * width,
            (np.arange(num_height) + 0.5) / num_height * height,
        ),
        -1,
    )
    return positions.reshape(-1, 2)


def _get_observations(env: BASEnv, positions: np.ndarray) -> np.ndarray:
    observations = []
    for position in positions:
        env.agent.position = position.astype(float)
        env.agent.velocity = (np.array([100,100])-position.astype(float))/1000
        observations.append(env.step(np.zeros((2,), dtype=float))[0])
    observations = np.array(observations)
    return observations


def _get_actions(
    model: PPO, observations: np.ndarray, num_samples: int = 100
) -> np.ndarray:
    actions = np.zeros((len(observations), 2), dtype=float)
    for _ in range(num_samples):
        actions += model.predict(observations)[0]
    return actions / num_samples


def _create_figure(
    env: BASEnv,
    positions: np.ndarray,
    actions: np.ndarray,
    dpi: int,
    arrow_scale: float,
) -> Figure:
    fig, ax = plt.subplots(dpi=dpi)
    # Move agent off screen for rendering.
    env.agent.position = env.blueprint.world_size * 2
    ax.imshow(env.render())

    ax.quiver(
        positions[:, 0] * env.window_scale,
        (env.blueprint.world_size[1] - positions[:, 1]) * env.window_scale,
        actions[:, 0],
        -actions[:, 1],
        angles="xy",
        scale_units="xy",
        scale=1 / env.window_scale / arrow_scale,
        pivot="mid",
    )
    ax.axis("off")

    ax.margins(0, 0)

    fig.tight_layout()

    return fig


def create_policy_plot(
    model_path: str,
    num_width: int,
    num_height: int,
    dpi: int,
    arrow_scale: float,
    save_path: pathlib.Path | None,
):
    """Plot or save policy for static environment."""

    # TODO: Make configurable
    env = GoalInsideGridEnv()
    env.reset()

    model = PPO.load(model_path)

    positions = _get_positions(env, num_width, num_height)
    observations = _get_observations(env, positions)
    actions = _get_actions(model, observations)

    fig = _create_figure(env, positions, actions, dpi, arrow_scale)

    if save_path is None:
        fig.show()
        plt.show()
    else:
        fig.savefig(save_path, pad_inches=0, bbox_inches="tight")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str)
    parser.add_argument("--num_width", type=int, default=50)
    parser.add_argument("--num_height", type=int, default=50)
    parser.add_argument(
        "--dpi",
        type=int,
        default=400,
    )
    parser.add_argument("--arrow_scale", type=float, default=4)
    parser.add_argument("--save_path", type=pathlib.Path)
    args = parser.parse_args()

    create_policy_plot(
        args.model_path,
        args.num_width,
        args.num_height,
        args.dpi,
        args.arrow_scale,
        args.save_path,
    )
