from typing import Any

import gymnasium as gym
import numpy as np


class BoidCollisionWrapper(gym.Wrapper):
    """BAS wrapper to modify reward and terminated based on a collision between agent and boids."""

    def __init__(
        self,
        env: gym.Env,
        collision_termination: bool = True,
        collision_reward: float | None = None,
        add_reward: bool = False,
    ):
        """Initialize the wrapper and modify reward range.

        Args:
            env: (Wrapped) BAS environment.
            collision_reward: Reward for a collision
            collision_termination: Terminate episode at collision.
            add_reward: Add collision reward to existing reward. Defaults to False.
        """
        super().__init__(env)

        self._collision_reward = collision_reward
        self._collision_termination = collision_termination
        self._add_reward = add_reward

        if collision_reward is not None:
            self._reward_range = min(self.reward_range[0], collision_reward), max(
                self.reward_range[1], collision_reward
            )

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict]:
        """Modify reward and terminated."""
        observation, reward, terminated, truncated, info = super().step(action)

        # TODO: Add functionality to compute these fundamental values in the base Env.
        #       Multiple wrappers can use the values then without re-computing.
        # Compute differences and distances to all boids.
        differences = self.swarm.positions - self.agent.position
        distances = np.linalg.norm(differences, axis=1)

        # Check for collision.
        if np.any(distances < self.swarm.radius + self.agent.radius):
            # Alter reward.
            if self._collision_reward is not None:
                reward = self._collision_reward + self._add_reward * reward
            # Set terminated.
            if self._collision_termination:
                terminated = True

        return observation, reward, terminated, truncated, info
