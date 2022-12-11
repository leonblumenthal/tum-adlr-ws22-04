from dataclasses import dataclass

import numpy as np
import pygame
from gymnasium import spaces

from swarm.agent import Agent, AgentConfig
from swarm.base_env import BaseEnv
from swarm.swarm import Swarm, SwarmConfig


@dataclass
class SwarmAndAgentEnvConfig:
    """Configurable parameters for a swarm and egent environment."""

    # [0, 100] x [0, 100]
    world_size = np.array([100, 100])

    obs_num_sections = 8
    obs_max_range = 20

    agent_config = AgentConfig(
        radius=1,
        speed=1,
    )

    swarm_config = SwarmConfig(
        num_boids=100,
        boid_radius=1,
        boid_max_speed=1,
        boid_max_acceleration=0.1,
        neighbor_range=10,
        seperation_range_fraction=0.5,
        steering_weights=(1.1, 1, 1),
        obstacle_margin=3,
    )


# TODO: I do not like the mixture between rendering and simulation.
#       Note, we never use images as observations.
class SwarmAndAgentEnv(BaseEnv):
    """Environment with a swarm and an agent."""

    # Types for items of observation, information, and action spaces.
    ObsType = np.ndarray
    InfoType = dict[str]
    ActType = np.ndarray

    def __init__(
        self,
        config: SwarmAndAgentEnvConfig = SwarmAndAgentEnvConfig(),
        render_mode: str | None = None,
        window_scale: float = 10,
    ):
        """Initialize the environment instance.

        Args:
            config: Configuration for environment, agent, and swarm parameters.
            render_mode: Render mode of the environment. Defaults to None.
            window_scale: Scaling factor for rendering the world. The window and frame size will be `world_size` * `window_scale`. Defaults to 10.
        """

        super().__init__(
            render_mode,
            window_size=(
                config.world_size[0] * window_scale,
                config.world_size[1] * window_scale,
            ),
        )

        self.world_size = config.world_size

        self.obs_num_sections = config.obs_num_sections
        self.obs_max_range = config.obs_max_range

        self.agent = Agent(config.agent_config, self.world_size, self.np_random)

        self.swarm = Swarm(config.swarm_config, self.world_size, self.np_random)

        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(self.obs_num_sections, 2), dtype=np.float
        )

        self.action_space = spaces.Box(low=-1, high=1, shape=(2,))

        # Window size is scaled version of world.
        self.window_scale = window_scale
        self.colors = dict(
            background="white",
            agent="blue",
            boid="red",
            boid_direction="black",
            observation_section="gray",
            observation="green",
        )

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[ObsType, InfoType]:
        """Reset and intialize the state of the environment.

        State related variables, e.g. location of the agent, are defined here.
        If the render mode is "human", the inital frame is also rendered.

        Args:
            seed: Seed for the internal RNG `self.random`. Defaults to None.
            options: Additional options. Defaults to None.

        Returns:
            Observation and information of the initial environment's state.
        """
        super().reset(seed=seed)

        self.step_counter = 0

        self.agent.reset()
        self.swarm.reset()

        observation = self._get_observation()
        info = self._get_info()

        # Render to window.
        if self.render_mode == "human":
            self.render_human()

        return observation, info

    def step(self, action: ActType) -> tuple[ObsType, float, bool, bool, dict]:
        """Update the environment's state based on the specified action.

        Args:
            action: action of the agent

        Returns:
            Observation, reward, terminated, truncated, and information after the update step.
        """

        self.step_counter += 1

        self.agent.step(action)
        self.swarm.step()

        # Render to window.
        if self.render_mode == "human":
            self.render_human()

        observation = self._get_observation()
        # TODO: I do not like this cache thing.
        #       Refer to _get_observation
        reward = (self._cache["distances"] < self.obs_max_range).sum()
        terminated = False
        truncated = False
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def _get_observation(self) -> ObsType:
        """Return the scaled relative positions of each nearest neighbor per section."""

        differences = self.swarm.boid_positions - self.agent.location
        distances = np.linalg.norm(differences, axis=1)
        angles = np.arctan2(differences[:, 1], differences[:, 0]) % (2 * np.pi)
        sections = np.floor_divide(angles, 2 * np.pi / self.obs_num_sections).astype(
            int
        )

        indices = np.arange(len(sections))

        observation = np.zeros((self.obs_num_sections, 2))

        for section in range(self.obs_num_sections):
            mask = sections == section

            if mask.sum() > 0:
                i = np.argmin(distances[mask])
                i = indices[mask][i]

                if distances[i] < self.obs_max_range:
                    observation[section] = differences[i] / self.obs_max_range
                    continue

            # Set default observation
            angle = (section + 0.5) * 2 * np.pi / self.obs_num_sections
            observation[section] = np.array([np.cos(angle), np.sin(angle)])

        # TODO: I do not like this cache thing.
        #       However, explictely passing everything is cumbersome.
        self._cache = dict(
            observation=observation,
            distances=distances,
        )

        return observation

    def _get_info(self) -> InfoType:
        """Get auxiliary information for the current state of the environment."""
        return {"step": self.step_counter}

    def render_canvas(self) -> pygame.Surface:
        """Actually render and return the canvas.

        The window size in "human" mode and frame size in "rgb_array" mode is equal.
        """

        # Canvas with background color.
        canvas = pygame.Surface(self.window_size)
        canvas.fill(self.colors["background"])

        # Draw agent.
        pygame.draw.circle(
            canvas,
            self.colors["agent"],
            self.agent.location * self.window_scale,
            self.agent.radius * self.window_scale,
        )

        # Draw each boid in swarm.
        for position, velocity in zip(
            self.swarm.boid_positions, self.swarm.boid_velocities
        ):
            pygame.draw.circle(
                canvas,
                self.colors["boid"],
                position * self.window_scale,
                self.swarm.boid_radius * self.window_scale,
            )
            pygame.draw.line(
                canvas,
                self.colors["boid_direction"],
                position * self.window_scale,
                (position + velocity) * self.window_scale,
                2,
            )

        # TODO: Only render when e.g. "debug" is enabled.
        for i, pos in enumerate(self._cache["observation"]):
            # Draw line to observation.
            pygame.draw.line(
                canvas,
                # TODO: colors
                self.colors["observation"],
                self.agent.location * self.window_scale,
                (self.agent.location + pos * self.obs_max_range) * self.window_scale,
                2,
            )
            # Draw observation circles.
            pygame.draw.circle(
                canvas,
                self.colors["observation"],
                (self.agent.location + pos * self.obs_max_range) * self.window_scale,
                self.swarm.boid_radius / 2 * self.window_scale,
            )

            # Draw section lines.
            angle = i * 2 * np.pi / self.obs_num_sections
            pygame.draw.line(
                canvas,
                self.colors["observation_section"],
                self.agent.location * self.window_scale,
                (
                    self.agent.location
                    + np.array([np.cos(angle), np.sin(angle)]) * self.obs_max_range
                )
                * self.window_scale,
                1,
            )

        # Ensure that lower left corner is (0,0).
        canvas = pygame.transform.flip(canvas, flip_x=False, flip_y=True)

        return canvas
