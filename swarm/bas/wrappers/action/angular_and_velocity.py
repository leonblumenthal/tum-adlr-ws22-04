import gymnasium as gym
import numpy as np


class AngularAndVelocityActionWrapper(gym.ActionWrapper):
    def __init__(self, env: gym.Env, max_steering_angle: float = np.pi):
        super().__init__(env)

        self._max_steering_angle = max_steering_angle

        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,))

    def action(self, action: np.ndarray) -> np.ndarray:
        angle = action[0] * self._max_steering_angle + self.agent.angle
        direction = np.array([np.cos(angle), np.sin(angle)])
        direction *= (action[1] + 1) / 2

        return direction
