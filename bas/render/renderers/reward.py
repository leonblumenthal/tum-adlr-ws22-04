import numpy as np
import pygame

from bas import wrappers
from bas.render.constants import Colors
from bas.render.renderers.renderer import Renderer


class TargetRewardWrapperRenderer(Renderer):
    """Renderer for the DistanceToTargetRewardWrapper drawing the target."""

    WRAPPER = wrappers.TargetRewardWrapper

    def render(self, canvas: pygame.Surface):
        """Draw the target position."""
        self.circle(canvas, Colors.REWARD, self._position, self._target_radius)


class NumNeighborsRewardWrapperRenderer(Renderer):
    """Renderer for the NumNeighborsRewardWrapper drawing neighbors."""

    WRAPPER = wrappers.NumNeighborsRewardWrapper

    def render(self, canvas: pygame.Surface):
        """Draw the neighbors."""

        # TODO: Add fundamental compuatations cache in env.
        distances = np.linalg.norm(self.swarm.positions - self.agent.position, axis=1)
        neighbors = self.swarm.positions[distances <= self._max_range]
        for position in neighbors:
            self.circle(canvas, Colors.REWARD, position, self.swarm.config.radius)

        self.circle(canvas, Colors.REWARD, self.agent.position, self._max_range, 2)
