import numpy as np
import pygame

from swarm.bas import wrappers
from swarm.bas.render.constants import Colors
from swarm.bas.render.renderers.renderer import Renderer


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

        for i, pos in enumerate(self.cached_step.observation):
            # TODO: Handle target direction differently.
            if i < len(self.cached_step.observation) - 1:
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
            else:
                # Draw line to observation.
                self.line(
                    canvas,
                    Colors.OBSERVATION,
                    agent.position,
                    agent.position + pos * max_range * 1.2,
                    4,
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

class SectionAndVelocityObservationWrapperRenderer(Renderer):
    """Renderer for the SectionObservationWrapper drawing observations and sections."""

    WRAPPER = wrappers.SectionAndVelocityObservationWrapper

    def render(self, canvas: pygame.Surface):
        """Draw observations small circles and lines to the agent.
        Also draw sections with static lines from the agent to the max range.
        """
        agent = self.agent
        swarm = self.swarm
        num_sections = self._num_sections
        max_range = self._max_range
        section_observation = self.cached_observation[:-1]
        velocity = self.cached_observation[-1]

        for i, pos in enumerate(section_observation):
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

            # Draw velocity line.
            self.line(
                canvas,
                Colors.AGENT,
                agent.position,
                agent.position + velocity * max_range,
                2,
            )
            self.circle(
                canvas,
                Colors.AGENT,
                agent.position + velocity * max_range,
                0.2,
            )

class DistanceSectionObservationWrapperRenderer(Renderer):
    """Renderer for the DistanceSectionObservationWrapper drawing observations and sections."""

    WRAPPER = wrappers.DistanceSectionObservationWrapper

    def render(self, canvas: pygame.Surface):
        """Draw observed distances and sections."""
        agent = self.agent
        num_sections = self._num_sections
        max_range = self._max_range

        angle_step = 2 * np.pi / num_sections

        for i, distance in enumerate(self.cached_step.observation):
            # Draw section lines.
            angle = i * angle_step
            self.line(
                canvas,
                Colors.MISC,
                agent.position,
                agent.position + np.array([np.cos(angle), np.sin(angle)]) * max_range,
                1,
            )
            self.line(
                canvas,
                Colors.OBSERVATION,
                agent.position,
                agent.position
                + np.array(
                    [np.cos(angle + angle_step / 2), np.sin(angle + angle_step / 2)]
                )
                * distance,
                2,
            )


class TargetDirectionObservationWrapperRenderer(Renderer):
    """Renderer for the TargetDirectionObservationWrapper drawing the direction to the target."""

    WRAPPER = wrappers.TargetDirectionObservationWrapper

    def render(self, canvas: pygame.Surface):
        """Draw target direction."""

        self.line(
            canvas,
            Colors.OBSERVATION,
            self.agent.position,
            self.agent.position + self.cached_step.observation * self.agent.radius * 2,
            4,
        )


class TargetDirectionAndSectionObservationWrapperRenderer(
    SectionObservationWrapperRenderer
):
    WRAPPER = wrappers.TargetDirectionAndSectionObservationWrapper
