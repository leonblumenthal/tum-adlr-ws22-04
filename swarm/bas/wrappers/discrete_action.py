import gymnasium as gym
import numpy as np
from gymnasium import spaces


class DiscreteActionWrapper(gym.ActionWrapper):
    """BAS wrapper to map action from discrete to equally spaced direction vectors."""

    ActionType = int

    def __init__(
        self,
        env: gym.Env,
        num_actions: int,
        angle_offset: float = 0,
        zero_action: bool = True,
    ):
        """Initilaize the wrapper and set the action space.

        Args:
            env: (Wrapped) BAS environment.
            num_actions: Number of equally spaced direction vectors (+1 if zero_action).
            angle_offset: Rotational offset for the direction vectors. Defaults to 0.
            zero_action: Add action zero to zero vector.. Defaults to True.
        """
        super().__init__(env)

        if zero_action:
            self.action_to_direction = [np.zeros(2)]
            num_actions -= 1

        angles = np.arange(num_actions) / num_actions * 2 * np.pi + angle_offset
        self.action_to_direction += [
            np.array([np.cos(angle), np.sin(angle)]) for angle in angles
        ]

        self._action_space = spaces.Discrete(len(self.action_to_direction))

    def action(self, action: ActionType) -> np.ndarray:
        """Map discrete action to direction vector."""
        return self.action_to_direction[action]
