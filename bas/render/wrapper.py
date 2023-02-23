from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
import pygame

from bas.render.constants import Colors
from bas.render.renderers.bas import BASEnvRenderer
from bas.render.renderers.misc import AgentTrajectoryRenderer
from bas.render.renderers.observation import \
    ObservationContainerWrapperRenderer
from bas.render.renderers.renderer import Renderer
from bas.render.renderers.reward import (
    NumNeighborsRewardWrapperRenderer, TargetRewardWrapperRenderer)


@dataclass
class CachedStep:
    action: Any
    observation: list[np.ndarray]
    reward: float
    terminated: bool
    truncated: bool
    info: dict


class RenderWrapper(gym.Wrapper):
    """Wrapper to render (wrapped) a BAS environments in "rgb_array" mode.

    Without this, the a BAS environment does not support rendering.

    For all nested envs/wrappers execute all applicable renderers in `enabled_renderers`.

    This wrapper should be above other custom BAS-related wrappers.
    """

    def __init__(
        self,
        env: gym.Env,
        enabled_renderers: list[type[Renderer]] = [
            BASEnvRenderer,
            TargetRewardWrapperRenderer,
            NumNeighborsRewardWrapperRenderer,
            AgentTrajectoryRenderer,
            ObservationContainerWrapperRenderer,
        ],
        window_scale: float = 10,
        return_numpy: bool = True,
    ):
        """Initialize the wrapper and setup all enabled and applicable renderers.

        Args:
            env: (Wrapped) BAS environment to render.
            enabled_renderers: Enabled renderers that are used if applicable.
            window_scale: Ratio between window size and world size. Defaults to 10.
            return_numpy: Return numpy array instead of PyGame canvas. Defaults to True.
        """
        super().__init__(env)

        # Attribute access is passed thorugh the "chain" of envs.
        self.window_scale = window_scale
        self.window_size = self.env.blueprint.world_size * window_scale
        self._renderers = self._get_applicable_renderers(enabled_renderers)

        self._metadata = env.metadata
        self._metadata["render_fps"] = 30

        self._return_numpy = return_numpy

    def _get_applicable_renderers(
        self, enabled_renderers: list[Renderer]
    ) -> list[Renderer]:
        """Find and create all applicable renderers based on all nested wrappers/envs."""
        # Create mapping from wrapper/env to list of renderers.
        wrapper_to_renderers = {}
        for renderer in enabled_renderers:
            if renderer.WRAPPER not in wrapper_to_renderers:
                wrapper_to_renderers[renderer.WRAPPER] = []
            wrapper_to_renderers[renderer.WRAPPER].append(renderer)

        # Instantiate corresponding renderers for all nested wrappers/env.
        renderers = []
        instance = self.env
        while instance is not None:
            for Renderer in wrapper_to_renderers.get(type(instance), []):
                renderers.append(Renderer(self, instance))

            instance = getattr(instance, "env", None)

        # Reverse renderers to draw root env first.
        return renderers[::-1]

    @property
    def render_mode(self) -> str:
        return "rgb_array"

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict]:
        """Forward step function but cache step results."""
        ret = super().step(action)

        # Cache for rendering.
        self.cached_step = CachedStep(action, *ret)

        return ret

    def reset(self, **kwargs) -> tuple[Any, dict]:
        """Forward reset function but cache reset results."""
        ret = super().reset(**kwargs)

        # Cache for rendering.
        self.cached_step = CachedStep(None, ret[0], 0, False, False, {})

        return ret

    def render(self) -> np.ndarray:
        """Create canvas and let all renderers draw on it starting from the BAS environment."""

        # Canvas with background color.
        canvas = pygame.Surface(self.window_size)
        canvas.fill(Colors.BACKGROUND)

        # Let all renderes draw on the canvas.
        for renderer in self._renderers:
            renderer.render(canvas)

        # Ensure that lower left corner is (0,0).
        canvas = pygame.transform.flip(canvas, flip_x=False, flip_y=True)

        if not self._return_numpy:
            return canvas

        return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))
