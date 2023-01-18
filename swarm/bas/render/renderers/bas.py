import pygame

from swarm.bas import Agent, BASEnv, Swarm
from swarm.bas.render.constants import Colors
from swarm.bas.render.renderers.renderer import Renderer


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
