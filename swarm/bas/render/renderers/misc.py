import gymnasium as gym
import pygame

from swarm.bas import BASEnv
from swarm.bas.render.constants import Colors
from swarm.bas.render.renderers.renderer import Renderer
from swarm.bas.wrappers import TrajectoryWrapper


class AgentTrajectoryRenderer(Renderer):
    """Renderer for rendering trajectories of the agent."""

    WRAPPER = TrajectoryWrapper

    def __init__(self, render_wrapper: "RenderWrapper", instance: gym.Wrapper):
        super().__init__(render_wrapper, instance)

        self._positions = [[]]

    def render(self, canvas: pygame.Surface):
        """Draw and store the trajectories of the agent."""

        self._positions[-1]=self._trajectory
        if self.cached_step.terminated or self.cached_step.truncated:
            self._positions.append([])
        for positions in self._positions:
            for a, b in zip(positions, positions[1:]):
                self.line(canvas, Colors.AGENT, a, b, 1)
