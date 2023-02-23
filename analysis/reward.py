import numpy as np

from analysis import utils
from bas import BASEnv


def _get_rewards(
    env: BASEnv, positions: np.ndarray, velocities: np.ndarray
) -> np.ndarray:
    rewards = []
    for position, velocity in zip(positions, velocities):
        env.agent.position = position.astype(float)
        env.agent.velocity = velocity
        rewards.append(env.step(np.zeros((2,), dtype=float))[1])

    return np.array(rewards)


def create_reward_heatmap(
    env: BASEnv,
    agent_velocity: np.ndarray | None = np.zeros(2),
    agent_velocity_target: np.ndarray | None = None,
    num_width: int = 100,
    num_height: int = 100,
):
    """Create a heatmap of the rewards for a static environment.

    The agent either has a fixed velocity or is pointing at a target.
    """

    env.reset()

    positions = utils.generate_grid_positions(env, num_width, num_height)
    velocities = utils.generate_velocities(
        positions, agent_velocity, agent_velocity_target
    )
    rewards = _get_rewards(env, positions, velocities)

    heatmap = rewards.reshape((num_height, num_width), order="C")[::-1]

    return heatmap
