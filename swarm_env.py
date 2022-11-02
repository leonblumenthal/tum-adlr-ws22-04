from typing import Iterator, Optional, Union
import gymnasium as gym
import numpy as np
from dataclasses import dataclass
import pygame
from gymnasium import spaces
import time
import abc

# TODOL: represent as all boids as single matrix
@dataclass
class Boid:
    position: np.ndarray
    velocity: np.ndarray


class SwarmNaive:
    def __init__(
        self,
        num_boids: int,
        world_size: float,
        np_random: np.random,
        neighbor_distance: float = 5,
    ):
        self._boids = [
            Boid(np_random.uniform(0, world_size, (2,)), np.zeros((2,)))
            for _ in range(num_boids)
        ]
        self._world_size = world_size
        self._np_random = np_random
        self._neighbor_distance = neighbor_distance

    def step(self):

        for boid in self._boids:
            neighbors = self._get_neighbors(boid)
            velocity = self._compute_velocitiy(boid, neighbors)
            boid.velocity = velocity

        for boid in self._boids:
            boid.position += boid.velocity

    def _get_neighbors(self, boid: Boid) -> list[Boid]:
        return [
            n
            for n in self._boids
            if n is not boid
            and np.linalg.norm(boid.position - n.position) < self._neighbor_distance
        ]

    def _compute_velocitiy(self, boid: Boid, neighbors: list[Boid]) -> np.ndarray:
        velocity = self._np_random.uniform(-1, 1, (2,)) / 2

        if not neighbors:
            return velocity

        cohesion = np.mean([n.position for n in neighbors], axis=0) - boid.position
        cohesion /= np.linalg.norm(cohesion)

        alignment = (
            np.mean([n.velocity for n in neighbors], axis=0)
            + self._np_random.uniform(-1, 1, (2,)) / 20
        )
        alignment /= np.linalg.norm(alignment)

        separation = np.sum(
            [
                (boid.position - n.position)
                * (
                    1
                    - np.linalg.norm(boid.position - n.position)
                    / self._neighbor_distance
                )
                ** 2
                for n in neighbors
            ],
            axis=0,
        )
        separation /= np.linalg.norm(separation)

        velocity += alignment + cohesion + separation

        bounding = np.zeros(
            2,
        )
        x, y = boid.position
        margin = 10
        if x < margin:
            bounding[0] = 1 - x / margin
        elif x > self._world_size - margin:
            bounding[0] = -(1 - (self._world_size - x) / margin)

        if y < margin:
            bounding[1] = 1 - y / margin
        elif y > self._world_size - margin:
            bounding[1] = -(1 - (self._world_size - y) / margin)

        velocity /= np.linalg.norm(velocity)
        velocity += bounding

        return velocity

    def get_boid_positions(self) -> Iterator[np.ndarray]:
        for boid in self._boids:
            yield boid.position


class ParallelSwarm:
    def __init__(
        self,
        num_boids: int,
        world_size: float,
        np_random: np.random,
        neighbor_distance: float = 5,
    ):
        self._boid_positions = np_random.random((num_boids, 2)) * world_size
        self._boid_velocities = np.zeros_like(self._boid_positions)
        self._world_size = world_size
        self._np_random = np_random
        self._neighbor_distance = neighbor_distance

    def step(self):
        deltas = self._boid_positions[None, :, :] - self._boid_positions[:, None, :]
        distances = (
            np.linalg.norm(deltas, axis=-1)
            + np.eye(deltas.shape[0]) * self._neighbor_distance
        )

        adjacency = distances < self._neighbor_distance

        cohesion = np.sum(deltas * adjacency[..., None], axis=1)
        cohesion_norm = np.linalg.norm(cohesion, axis=-1)
        cohesion[cohesion_norm > 0] /= cohesion_norm[cohesion_norm > 0][:, None]

        alignment = np.sum(
            self._boid_velocities[None, ...] * adjacency[..., None], axis=1
        )
        alignment_norm = np.linalg.norm(alignment, axis=-1)
        alignment[alignment_norm > 0] /= alignment_norm[alignment_norm > 0][:, None]

        separation = -np.sum(
            deltas
            * np.clip(1 - distances / (self._neighbor_distance / 5), 0, None)[..., None]
            ** 2,
            axis=1,
        )
        separation_norm = np.linalg.norm(separation, axis=-1)
        separation[separation_norm > 0] /= separation_norm[separation_norm > 0][:, None]

        self._boid_velocities = 3 * alignment + cohesion/2 + separation


        


        self._boid_positions += self._boid_velocities / 5

    def get_boid_positions(self) -> Iterator[np.ndarray]:
        for pos in self._boid_positions:
            yield pos


class SwarmEnv(gym.Env):
    """
    https://gymnasium.farama.org/tutorials/environment_creation/
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        size: float = 100,
    ):
        super().__init__()

        assert render_mode in [None, *self.metadata["render_modes"]]
        self.render_mode = render_mode

        self.size = size

        self.observation_space = spaces.Box(
            low=0, high=self.size, shape=(2,), dtype=np.float
        )

        self.action_space = spaces.Discrete(4)
        self._action_to_direction = {
            0: np.array([0, 0]),  # stay
            1: np.array([1, 0]),  # right
            2: np.array([0, -1]),  # top (negative y)
            3: np.array([-1, 0]),  # left
            4: np.array([0, 1]),  # down
        }

        self.window = None
        self.window_size = 1000
        self.clock = None

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> tuple["ObsType", dict]:
        super().reset(seed=seed)

        self._agent_location = self.np_random.random(size=2) * self.size

        self._swarm = ParallelSwarm(100, self.size, self.np_random, 20)

        if self.render_mode == "human":
            self._render_frame()

        observation = self._agent_location
        info = {}

        return observation, info

    def step(self, action: "ActType") -> tuple["ObsType", float, bool, bool, dict]:

        self._agent_location += self._action_to_direction[action]
        self._agent_location = np.clip(self._agent_location, 0, self.size)

        # Update swarm.
        start = time.perf_counter()
        self._swarm.step()
        print("step", time.perf_counter() - start)

        if self.render_mode == "human":
            self._render_frame()

        observation = self._agent_location
        reward = 0
        terminated = False
        truncated = False
        info = {}

        return observation, reward, terminated, truncated, info

    def render(self) -> Optional[Union["RenderFrame", list["RenderFrame"]]]:
        if self.render_mode == "rgb_array":
            return self._render_frame()

    # NOTE: Mainly copied from tutorial
    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))

        world_to_screen = self.window_size / self.size

        # Draw agent.
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            self._agent_location * world_to_screen,
            2 * world_to_screen,
        )
        # Draw swarm.
        # TODO: Optimize
        start = time.perf_counter()
        for position in self._swarm.get_boid_positions():
            pygame.draw.circle(
                canvas,
                (255, 0, 0),
                position * world_to_screen,
                0.5 * world_to_screen,
            )
        print("drawing", time.perf_counter() - start)
        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
