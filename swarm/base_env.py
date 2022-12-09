import abc
from typing import TypeVar

import gymnasium as gym
import numpy as np
import pygame

# TODO: Possibly simplify using gymnasium.wrappers.HumanRendering.
class BaseEnv(gym.Env, abc.ABC):
    """Base class for all our Gymnasium environment that mainly hides repetitive rendering code."""

    # NOTE: These should be newly set for each sublcass.
    # Types for items of observation, information, and action spaces.
    ObsType = TypeVar("ObsType")
    InfoType = TypeVar("InfoType")
    ActType = TypeVar("ActType")

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode: str | None, window_size: tuple[float, float]):
        """Constructor of the `BaseEnv` class.

        Args:
            render_mode:  Render mode of the environment. Defaults to None.
        """

        assert render_mode in [None, *self.metadata["render_modes"]]
        self.render_mode = render_mode

        # Stuff needed for rendering in human mode.
        self.window_size = window_size
        self.window = None
        self.clock = None

    def render_human(self):
        """Render frame and show it in window.

        This is only used in "human" mode and called by the `reset` and `step` functions.
        """

        # Setup PyGame window once in the first call.
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(self.window_size)
        if self.clock is None:
            self.clock = pygame.time.Clock()

        # Actually render the environment.
        canvas = self.render_canvas()

        # Update window.
        self.window.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])

    def render(self) -> np.ndarray | None:
        """Render environment and return frame (only used for "rgb_array" mode).

        The frame size in "rgb_array" mode is also `self._window_size`.

        Rendering for "human" mode is handled by the `_render_frame` method
        which is called by the `reset` and `step` methods.
        """
        if self.render_mode == "rgb_array":
            canvas = self.render_canvas()
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    @abc.abstractmethod
    def render_canvas(self) -> pygame.Surface:
        """Actually render and return the canvas."""
        raise NotImplementedError
