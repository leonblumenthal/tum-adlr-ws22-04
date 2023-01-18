import abc
from typing import Any

import gymnasium as gym
import numpy as np
import pygame


class Renderer(abc.ABC):
    """Abstact base class for renderers that render BAS wrappers.

    `self.` should be used. to access public attributes of the render wrapper
    and private attributes of the corresponding `WRAPPER` instance.

    This class also contains bindings for pygame draw functions
    that automatically convert from world size to window size.
    """

    WRAPPER = None

    def __init__(self, render_wrapper: "RenderWrapper", instance: gym.Wrapper):
        """Initalize renderer and store wrappers.

        Args:
            render_wrapper: Render wrapper needed for window scale.
            instance: Actual `WRAPPER` instance corresponding to this renderer with private attributes.
        """
        self._render_wrapper = render_wrapper
        self._instance = instance

    def __getattr__(self, name: str) -> Any:
        """Provide access to public attributes of the render_wrapper
        and private attributes of the corresponding `WRAPPER` instance.
        """
        if name.startswith("_"):
            return getattr(self._instance, name)

        return getattr(self._render_wrapper, name)

    @abc.abstractmethod
    def render(self, canvas: pygame.Surface):
        """Draw additional stuff from the wrapper on the canvas."""

    def line(
        self,
        canvas: pygame.Surface,
        color: str,
        start_position: np.ndarray,
        end_position: np.ndarray,
        width: float,
    ):
        """Draw a line on the canvas in window size from geometric arguments in world size."""
        scale = self.window_scale
        pygame.draw.line(
            canvas, color, scale * start_position, scale * end_position, width
        )

    def circle(
        self,
        canvas: pygame.Surface,
        color: str,
        center: np.ndarray,
        radius: float,
        width: int = 0,
    ):
        """Draw a circle on the canvas in window size from geometric arguments in world size."""
        scale = self.window_scale
        pygame.draw.circle(canvas, color, scale * center, scale * radius, width)
