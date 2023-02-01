import gymnasium as gym
import numpy as np


class RelativeRotationWrapper(gym.ActionWrapper):
    """BAS wrapper for transforming relative (N,2) observations into the agent direction frame
    and relative (2,) actions into the world frame."""

    def reset(self, **kwargs):
        """Rotate observation from world into agent direction frame."""

        self._cached_rotation = None

        observation, info = self.env.reset(**kwargs)
        R = self._compute_rotation()
        return observation @ R.T, info

    def step(self, action):
        """Rotate action into world and observation into agent direction frame."""
        R = self._compute_rotation()
        observation, reward, terminated, truncated, info = self.env.step(R.T @ action)
        return observation @ R.T, reward, terminated, truncated, info

    def _compute_rotation(self) -> np.ndarray:
        """Compute rotation matrix from world into agent direction frame."""
        velocity_norm = np.linalg.norm(self.agent.velocity)

        # Return last rotation if velocity is zero.
        if velocity_norm < 1e-6:
            if self._cached_rotation is None:
                angle = self.np_random.random() * 2 * np.pi
                sin = np.sin(angle)
                cos = np.cos(angle)
                return np.array([[cos, sin], [-sin, cos]])
            return self._cached_rotation

        direction = self.agent.velocity / velocity_norm
        rotation = np.array([direction, [-direction[1], direction[0]]])
        self._cached_rotation = rotation

        return rotation
