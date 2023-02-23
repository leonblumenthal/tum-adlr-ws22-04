import numpy as np

from bas import BASEnv


def generate_grid_positions(env: BASEnv, num_width: int, num_height: int) -> np.ndarray:
    """Generate positions aranged in a grid based on the environment's world size"""
    width, height = env.blueprint.world_size
    positions = np.stack(
        np.meshgrid(
            (np.arange(num_width) + 0.5) / num_width * width,
            (np.arange(num_height) + 0.5) / num_height * height,
        ),
        -1,
    )
    return positions.reshape(-1, 2)

def generate_velocities(
    positions: np.ndarray,
    agent_velocity: np.ndarray | None,
    agent_velocity_target: np.ndarray | None,
) -> np.ndarray:
    """Generate either constant velocities or directed at a target for each position."""
    assert (agent_velocity is None) != (agent_velocity_target is None)

    if agent_velocity_target is None:
        velocities = np.tile(agent_velocity, (len(positions), 1))
    else:
        velocities = agent_velocity_target - positions
        velocities /= np.linalg.norm(velocities, axis=1, keepdims=True) + 1e-6
    return velocities