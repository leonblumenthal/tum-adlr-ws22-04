import gymnasium as gym
import numpy as np


class DesiredVelocityActionWrapper(gym.ActionWrapper):
    """Rotate a 2D direction from the agent frame into the env frame and limit the norm to 1."""

    def __init__(self, env: gym.Env, in_agent_frame: bool = True):
        super().__init__(env)

        self._in_agent_frame = in_agent_frame

        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,))

    def action(self, action: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(action)
        if norm > 1:
            action = action / norm

        if not self._in_agent_frame:
            return action

        agent_to_env_rotation = np.array(
            [
                [np.cos(self.env.agent.angle), -np.sin(self.env.agent.angle)],
                [np.sin(self.env.agent.angle), np.cos(self.env.agent.angle)],
            ]
        )

        return agent_to_env_rotation @ action
