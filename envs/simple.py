import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pygame


class SimpleEnv(gym.Env):
    """Simple environment with a single agent."""

    # NOTE: This is needed for every custom environment.
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    # Types for items of observation, information, and action spaces.
    ObsType = dict[str, np.array]
    InfoType = dict[str]
    ActType = int

    def __init__(
        self,
        render_mode: str | None = None,
        world_size: float | tuple[float, float] = (100, 100),
        agent_radius: float = 1,
        agent_speed: float = 1,
        window_scale: float = 10,
    ):
        """Initialize the environment instance.

        State related variables such as the agent's location are defined inside the `reset` method.

        Args:
            render_mode: Render mode of the environment. Defaults to None.
            world_size: Size of the simulated environment. The lower left corner is (0,0). Defaults to (100, 100).
            agent_radius: Radius of the agent in the world. Defaults to 2.
            agent_speed: Velocity of the agent's actions. Defaults to 1.
            window_scale: Scaling factor for rendering the world. The window and frame size will be `world_size` * `window_scale`. Defaults to 10.
        """

        assert render_mode in [None, *self.metadata["render_modes"]]
        self.render_mode = render_mode

        # World is [0, world_size[0]] x [0, world_size[1]].
        self._world_size = (
            (world_size, world_size) if isinstance(world_size, float) else world_size
        )
        self._world_size = np.array(self._world_size)

        self._agent_radius = agent_radius
        self._agent_speed = agent_speed

        # Position of the agent is the only observation for now.
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(
                    low=agent_radius,
                    high=self._world_size - agent_radius,
                    dtype=np.float,
                )
            }
        )

        # 5 possible actions (1 nothing + 4 directions).
        self.action_space = spaces.Discrete(5)
        # Translation between dsicrete actions and directions in the environment.
        self._action_to_direction = {
            0: np.array([0, 0]),  # stay
            1: np.array([1, 0]),  # right
            2: np.array([0, 1]),  # top
            3: np.array([-1, 0]),  # left
            4: np.array([0, -1]),  # down
        }

        # Window size is scaled version of world.
        self.window_size = (
            self._world_size[0] * window_scale,
            self._world_size[1] * window_scale,
        )
        self._window_scale = window_scale
        self._colors = {"background": (255, 255, 255), "agent": (0, 0, 255)}
        # Stuff needed for rendering in human mode.
        self.window = None
        self.clock = None

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
        self._agent_location = (
            self.np_random.uniform(low=0.1, high=0.9, size=2) * self._world_size
        )

        # Render to window.
        if self.render_mode == "human":
            self._render_human()

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
        self._agent_location += self._action_to_direction[action] * self._agent_speed
        self._agent_location = np.clip(
            self._agent_location,
            a_min=self._agent_radius,
            a_max=self._world_size - self._agent_radius,
        )

        # Render to window.
        if self.render_mode == "human":
            self._render_human()

        observation = self._get_observation()
        reward = 0
        terminated = False
        truncated = False
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def _get_observation(self) -> ObsType:
        """Translate the current state of the environment into an observation."""
        return {"agent": self._agent_location}

    def _get_info(self) -> InfoType:
        """Get auxiliary information for the current state of the environment."""
        return {}

    def _render_human(self):
        """Render frame and show it in window.

        This is only used in "human" mode and called by the `reset` and `step` functions.
        """

        # Setup PyGame window once in the first call.
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        # Actually render the environment.
        canvas = self._render_canvas()

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
            canvas = self._render_canvas()
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def _render_canvas(self) -> pygame.Surface:
        """Actually render and return the canvas.

        The window size in "human" mode and frame size in "rgb_array" mode is equal.
        """

        # Canvas with background color.
        canvas = pygame.Surface(self.window_size)
        canvas.fill(self._colors["background"])

        # Draw agent.
        pygame.draw.circle(
            canvas,
            self._colors["agent"],
            self._agent_location * self._window_scale,
            self._agent_radius * self._window_scale,
        )

        # Ensure that lower left corner is (0,0).
        canvas = pygame.transform.flip(canvas, flip_x=False, flip_y=True)

        return canvas
