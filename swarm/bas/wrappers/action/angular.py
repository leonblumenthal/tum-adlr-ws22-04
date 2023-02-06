import gymnasium as gym
import numpy as np


class AngularActionWrapper(gym.ActionWrapper):
    def __init__(self, env: gym.Env, max_steering_angle: float = np.pi):
        super().__init__(env)

        self._max_steering_angle = max_steering_angle

        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=float)

    def action(self, action: np.ndarray) -> np.ndarray:
        # angle = (action * self._max_steering_angle).item() + self.agent.angle
        # direction = np.array([np.cos(angle), np.sin(angle)])

        # return direction
        return action
