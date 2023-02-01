import sys

import gymnasium
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

sys.modules["gym"] = gymnasium
from stable_baselines3 import PPO

from swarm.analysis import utils
from swarm.bas import BASEnv


def _get_observations(
    env: BASEnv, positions: np.ndarray, velocities: np.ndarray
) -> np.ndarray:
    observations = []
    for position, velocity in zip(positions, velocities):
        env.agent.position = position.astype(float)
        env.agent.velocity = velocity
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

    # Draw background image of the environment with agent off screen.
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


# TODO: THIS DOES NOT WORK WITH RELATIVE ENVIRONMENTS
def create_policy_plot(
    model: str | PPO,
    env: BASEnv,
    agent_velocity: np.ndarray | None = np.zeros(2),
    agent_velocity_target: np.ndarray | None = None,
    num_width: int = 50,
    num_height: int = 50,
    num_action_samples: int = 100,
    dpi: int = 200,
    arrow_scale: float = 4,
) -> Figure:
    """Plot action/policy of the model/agent for a static environment.

    The agent either has a fixed velocity or is pointing at a target.
    """

    if isinstance(model, str):
        model = PPO.load(model)

    env.reset()

    positions = utils.generate_grid_positions(env, num_width, num_height)
    velocities = utils.generate_velocities(
        positions, agent_velocity, agent_velocity_target
    )
    observations = _get_observations(env, positions, velocities)
    actions = _get_actions(model, observations, num_action_samples)

    figure = _create_figure(env, positions, actions, dpi, arrow_scale)

    return figure
