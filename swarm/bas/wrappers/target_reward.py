import gymnasium as gym
import numpy as np


class TargetRewardWrapper(gym.Wrapper):
    """BAS wrapper to compute reward based on distance to target."""

    def __init__(
        self,
        env: gym.Env,
        position: np.ndarray,
        distance_reward_scale: float = 1,
        target_radius: float = 3,
        target_reward: float = 100,
        target_termination: bool = True,
    ):
        """Initialize wrapper and set the reward range.

        Args:
            env: (Wrapped) BAS environment.
            position: Position of the target.
            distance_reward_scale: Scale of (1 - normalized distance between agent and target) reward
            target_radius: Distance at which agent has reached the target.
            target_reward: Reward for reaching the target, i.e. when inside the radius.
            target_termination: Terminate when reaching the target
        """
        super().__init__(env)

        self._position = position
        self._distance_reward_scale = distance_reward_scale
        self._target_radius = target_radius
        self._target_reward = target_reward
        self._target_termination = target_termination

        # TODO: Compute actual max distance possible based on target position.
        self._max_possible_distance = np.linalg.norm(self.blueprint.world_size)

        self._reward_range = (0, max(distance_reward_scale, target_reward))

    def step(self, action):
        """Set reward and termination based on distance between agent target."""
        observation, _, terminated, truncated, info = self.env.step(action)

        distance = np.linalg.norm(self.agent.position - self._position)
        if distance < self._target_radius:
            reward = self._target_reward
            terminated = terminated or self._target_termination
        else:
            normalized_distance = distance / self._max_possible_distance
            reward = (1 - normalized_distance) * self._distance_reward_scale

        return observation, reward, terminated, truncated, info
