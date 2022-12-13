import abc
from typing import Any

import gymnasium as gym
import numpy as np
import pygame

from swarm.bas import Agent, BASEnv, Swarm, wrappers
from swarm.bas.render.constants import Colors

# TODO: Split into multiple files.


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
    ):
        """Draw a circle on the canvas in window size from geometric arguments in world size."""
        scale = self.window_scale
        pygame.draw.circle(canvas, color, scale * center, scale * radius)


class BASEnvRenderer(Renderer):
    """Renderer for the raw BAS environment drawing the agent and the boids."""

    WRAPPER = BASEnv

    def render(self, canvas: pygame.Surface):
        """Draw the agent position and position and velocity of the boids."""
        agent: Agent = self.agent
        self.circle(canvas, Colors.AGENT, agent.position, agent.radius)

        swarm: Swarm = self.swarm
        for position, velocity in zip(swarm.positions, swarm.velocities):
            self.circle(canvas, Colors.BOID, position, swarm.radius)
            if swarm.max_velocity is not None:
                self.line(
                    canvas, Colors.BOID_DIRECTION, position, position + velocity, 2
                )


class DistanceToTargetRewardWrapperRenderer(Renderer):
    """Renderer for the DistanceToTargetRewardWrapper drawing the target."""

    WRAPPER = wrappers.DistanceToTargetRewardWrapper

    def render(self, canvas: pygame.Surface):
        """Draw the target position."""
        target_position = self._position

        self.circle(canvas, Colors.REWARD, target_position, 1)


class SectionObservationWrapperRenderer(Renderer):
    """Renderer for the SectionObservationWrapper drawing observations and sections."""

    WRAPPER = wrappers.SectionObservationWrapper

    def render(self, canvas: pygame.Surface):
        """Draw observations small circles and lines to the agent.
        Also draw sections with static lines from the agent to the max range.
        """
        agent = self.agent
        swarm = self.swarm
        num_sections = self._num_sections
        max_range = self._max_range

        for i, pos in enumerate(self.cached_observation):
            # Draw line to observation.
            self.line(
                canvas,
                Colors.OBSERVATION,
                agent.position,
                agent.position + pos * max_range,
                2,
            )
            # Draw observation circles.
            self.circle(
                canvas,
                Colors.OBSERVATION,
                agent.position + pos * max_range,
                swarm.radius / 2,
            )

            # Draw section lines.
            angle = i * 2 * np.pi / num_sections
            self.line(
                canvas,
                Colors.MISC,
                agent.position,
                agent.position + np.array([np.cos(angle), np.sin(angle)]) * max_range,
                1,
            )


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
            SectionObservationWrapperRenderer,
            DistanceToTargetRewardWrapperRenderer,
        ],
        window_scale: float = 10,
    ):
        """Initialize the wrapper and setup all enabled and applicable renderers.

        Args:
            env: (Wrapped) BAS environment to render.
            enabled_renderer: Enabled renderers that are used if applicable.
            window_scale: Ratio between window size and world size. Defaults to 10.
        """
        super().__init__(env)

        # Attributes access is passed thorugh the "chain" of envs.
        self.window_scale = window_scale
        self.window_size = self.env.blueprint.world_size * window_scale

        self._renderers = self._get_applicable_renderers(enabled_renderers)

    def _get_applicable_renderers(
        self, enabled_renderers: list[Renderer]
    ) -> list[Renderer]:
        """Find and create all applicable renderers based on all nested wrappers/envs."""
        wrapper_to_renderer = {
            renderer.WRAPPER: renderer for renderer in enabled_renderers
        }

        renderers = []
        instance = self.env
        while instance is not None:
            Renderer = wrapper_to_renderer.get(type(instance))
            if Renderer is not None:
                renderers.append(Renderer(self, instance))

            instance = getattr(instance, "env", None)

        # Reversed because root env should be drawn first.
        return renderers[::-1]

    @property
    def render_mode(self) -> str:
        return "rgb_array"

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict]:
        """Forward step function but cache observation and reward."""
        ret = super().step(action)

        # Cache observation and reward, for additional rendering.
        if self._renderers:
            self.cached_observation = ret[0]
            self.cached_reward = ret[1]

        return ret

    def reset(self, **kwargs) -> tuple[Any, dict]:
        """Forward reset function but cache observation and (0) reward."""
        ret = super().reset(**kwargs)

        # Cache observation and reward for additional rendering.
        if self._renderers:
            self.cached_observation = ret[0]
            self.cached_reward = 0

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

        return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))
