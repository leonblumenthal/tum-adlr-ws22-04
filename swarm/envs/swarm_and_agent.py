import numpy as np
import pygame
from gymnasium import spaces

from swarm.envs.base import BaseEnv
from swarm.swarm.swarm_2d import Swarm2D


class SwarmAndAgentEnv(BaseEnv):
    """Environment with a swarm and an agent."""

    # Types for items of observation, information, and action spaces.
    ObsType = dict[str, np.ndarray]
    InfoType = dict[str]
    ActType = int

    def __init__(
        self,
        render_mode: str | None = None,
        world_size: tuple[float, float] = (100, 100),
        agent_radius: float = 2,
        agent_speed: float = 1,
        window_scale: float = 10,
    ):
        """Initialize the environment instance.

        State related variables such as the agent's location are defined inside the `reset` method.

        Args:
            swarm: Swarm instance.
            render_mode: Render mode of the environment. Defaults to None.
            world_size: Size of the simulated environment. The lower left corner is (0,0). Defaults to (100, 100).
            agent_radius: Radius of the agent in the world. Defaults to 2.
            agent_speed: Velocity of the agent's actions. Defaults to 1.
            window_scale: Scaling factor for rendering the world. The window and frame size will be `world_size` * `window_scale`. Defaults to 10.
        """
        super().__init__(
            render_mode,
            window_size=(world_size[0] * window_scale, world_size[1] * window_scale),
        )

        # World is [0, world_size[0]] x [0, world_size[1]].
        self.world_size = (
            (world_size, world_size)
            if not isinstance(world_size, tuple)
            else world_size
        )
        self.world_size = np.array(self.world_size)

        self.agent_radius = agent_radius
        self.agent_speed = agent_speed

        # TODO: Make configurable.
        self.swarm = Swarm2D(
            num_boids=100,
            boid_radius=1,
            boid_max_speed=2,
            boid_max_acceleration=0.1,
            neighbor_range=10,
            seperation_range_fraction=0.5,
            steering_weights=(3, 2, 2),
            world_size=self.world_size,
            obstacle_margin=2,
            np_random=self.np_random,
        )

        # Position of the agent is the only observation for now.
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(
                    low=agent_radius,
                    high=self.world_size - agent_radius,
                    dtype=np.float,
                )
            }
        )

        # 5 possible actions (1 nothing + 4 directions).
        self.action_space = spaces.Discrete(5)
        # Translation between dsicrete actions and directions in the environment.
        self.action_to_direction = {
            0: np.array([0, 0]),  # stay
            1: np.array([1, 0]),  # right
            2: np.array([0, 1]),  # top
            3: np.array([-1, 0]),  # left
            4: np.array([0, -1]),  # down
        }

        # Window size is scaled version of world.
        self.window_scale = window_scale
        self.colors = {
            "background": (255, 255, 255),
            "agent": (0, 0, 255),
            "boid": (255, 0, 0),
        }

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

        # Sample random agent location.
        self.agent_location = (
            self.np_random.uniform(low=0.1, high=0.9, size=2) * self.world_size
        )

        self.swarm.reset()

        # Render to window.
        if self.render_mode == "human":
            self.render_human()

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(self, action: ActType) -> tuple[ObsType, float, bool, bool, dict]:
        """Update the environment's state based on the specified action.

        Args:
            action: action of the agent

        Returns:
            Observation, reward, terminated, truncated, and information after the update step.
        """

        # Update the agent's location based on action but stay inside the world.
        self.agent_location += self.action_to_direction[action] * self.agent_speed
        self.agent_location = np.clip(
            self.agent_location,
            a_min=self.agent_radius,
            a_max=self.world_size - self.agent_radius,
        )

        self.swarm.step()

        # Render to window.
        if self.render_mode == "human":
            self.render_human()

        observation = self._get_observation()
        reward = 0
        terminated = False
        truncated = False
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def _get_observation(self) -> ObsType:
        """Translate the current state of the environment into an observation."""
        return {"agent": self.agent_location}

    def _get_info(self) -> InfoType:
        """Get auxiliary information for the current state of the environment."""
        return {}

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
            self.agent_location * self.window_scale,
            self.agent_radius * self.window_scale,
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
                "black",
                position * self.window_scale,
                (position + velocity) * self.window_scale,
                2,
            )

        # Ensure that lower left corner is (0,0).
        canvas = pygame.transform.flip(canvas, flip_x=False, flip_y=True)

        return canvas
