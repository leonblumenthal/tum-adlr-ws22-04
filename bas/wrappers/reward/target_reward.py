from typing import Callable

import gymnasium as gym
import numpy as np


class TargetRewardWrapper(gym.Wrapper):
    """BAS wrapper to compute reward based on distance to target."""

    def __init__(
        self,
        env: gym.Env,
        position: np.ndarray,
        distance_reward_transform: Callable[[float], float] = lambda d: -d,
        target_radius: float = 3,
        target_reward: float = 1,
        target_termination: bool = True,
    ):
        """Initialize wrapper and set the reward range.

        Args:
            env: (Wrapped) BAS environment.
            position: Position of the target.
            distance_reward_transform: Callable to transform normalized distance between agent and target to reward
            target_radius: Distance at which agent has reached the target.
            target_reward: Reward for reaching the target, i.e. when inside the radius.
            target_termination: Terminate when reaching the target
        """
        super().__init__(env)

        self._position = position
        self._distance_reward_transform = distance_reward_transform
        self._target_radius = target_radius
        self._target_reward = target_reward
        self._target_termination = target_termination

        # TODO: Compute actual max distance possible based on target position.
        self._max_possible_distance = np.linalg.norm(self.blueprint.world_size)

        # Approximate reward range.
        reward_samples = [distance_reward_transform(i / 50) for i in range(51)]
        reward_samples.append(target_reward)
        self._reward_range = (min(reward_samples), max(reward_samples))

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        info["is_success"] = False
        return observation, info

    def step(self, action):
        """Set reward and termination based on distance between agent and target."""
        observation, _, terminated, truncated, info = self.env.step(action)
        info["is_success"] = False

        distance = np.linalg.norm(self.agent.position - self._position)
        if distance < self._target_radius:
            reward = self._target_reward
            terminated = terminated or self._target_termination
            info["is_success"] = True
        else:
            normalized_distance = distance / self._max_possible_distance
            reward = self._distance_reward_transform(normalized_distance)

        return observation, reward, terminated, truncated, info