import gymnasium as gym
import numpy as np


class AngularAndVelocityActionWrapper(gym.ActionWrapper):
    """Create a 2D desired velocity based on a steering angle relative the the agent direction"""

    def __init__(self, env: gym.Env, max_steering_angle: float = np.pi):
        super().__init__(env)

        self._max_steering_angle = max_steering_angle

        self.action_space = gym.spaces.Box(low=np.array((-1, 0)), high=np.array((1, 1)), shape=(2,))

    def action(self, action: np.ndarray) -> np.ndarray:
        steering, speed = action

        steering_angle = steering * self._max_steering_angle + self.agent.angle
        direction = np.array([np.cos(steering_angle), np.sin(steering_angle)]) * speed

        return direction
