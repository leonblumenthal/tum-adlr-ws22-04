import argparse
import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pygame

sys.modules["gym"] = gym


import numpy as np
from stable_baselines3 import PPO

from swarm.bas import wrappers
from swarm.bas.render.utils import inject_render_wrapper
from swarm.bas.wrappers import utils
from utils import parse_model_env_window_scale


@dataclass
class Dragable:
    position: np.ndarray
    velocity: np.ndarray
    radius: float


class Dragger:
    """Tool for interactive qualitative evaluation.

    Left click drags the positions of the boids, the agent, and the target.
    Right click drags their velocities.
    """

    POSITION_SELECTION = 1
    VELOCITY_SELECTION = 2

    def __init__(self, env: gym.Env, window_scale: float = 5) -> None:
        self.env = env

        self._init_env(window_scale)
        self._init_dragables()

        self.selected = None
        self.selection_mode = None

    def _init_env(self, window_scale: float):
        def _cache_action(action: np.ndarray):
            self._env_action = action

        self.env.agent.step = _cache_action
        self.env.swarm.step = lambda *_: None
        inject_render_wrapper(env, window_scale=window_scale, return_numpy=False)
        self.env.reset()

    def _init_dragables(self):
        # Boids
        self.dragables = [
            Dragable(p, v, env.swarm.config.radius)
            for p, v in zip(env.swarm._positions, env.swarm._velocities)
        ]
        # Target
        target_reward_wrapper = utils.get_wrapper(env, wrappers.TargetRewardWrapper)
        if target_reward_wrapper is not None:
            self.dragables.append(
                Dragable(
                    target_reward_wrapper._position,
                    np.array([0, 0]),
                    target_reward_wrapper._target_radius,
                )
            )
        # Agent.
        self.dragables.append(
            Dragable(env.agent.position, env.agent.velocity, env.agent.radius)
        )

    def run(self, model: PPO):
        pygame.init()
        screen = pygame.display.set_mode(self.env.window_size)
        clock = pygame.time.Clock()

        env_action = None

        is_running = True
        while is_running:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN and event.key in (
                    pygame.K_q,
                    pygame.K_ESCAPE,
                ):
                    is_running = False

                if event.type == pygame.MOUSEBUTTONDOWN:
                    for dragable in self.dragables:
                        if (
                            np.linalg.norm(
                                dragable.position - self._event_to_world(event.pos)
                            )
                            <= dragable.radius
                        ):
                            if event.button == 1:
                                self.selected = dragable
                                self.selection_mode = self.POSITION_SELECTION
                                break
                            elif event.button == 3:
                                self.selected = dragable
                                self.selection_mode = self.VELOCITY_SELECTION
                                dragable.velocity *= 0
                                break

                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button in (1, 3):
                        if self.selection_mode == self.VELOCITY_SELECTION:
                            print(
                                f"velocity norm: {np.linalg.norm(self.selected.velocity)}"
                            )
                        self.selected = None
                        self.selection_mode = None

                elif event.type == pygame.MOUSEMOTION:
                    if self.selected is not None:
                        if self.selection_mode == self.POSITION_SELECTION:
                            self.selected.position[:] = self._event_to_world(event.pos)
                        elif self.selection_mode == self.VELOCITY_SELECTION:
                            self.selected.velocity[:] = (
                                self._event_to_world(event.pos) - self.selected.position
                            ) / 10
                            # Agent
                            if self.selected is self.dragables[-1]:
                                self.env.agent.angle = np.arctan2(
                                    self.env.agent.velocity[1],
                                    self.env.agent.velocity[0],
                                )

            canvas = self.env.render()

            if self.selected is not None:
                observation, reward, terminated, truncated, info = self.env.step(
                    np.zeros(self.env.action_space.shape)
                )
                agent_action = model.predict(observation, deterministic=True)[0]
                self.env.step(agent_action)
                env_action = self._env_action

                print(f"{observation=}")
                print(f"{reward=}")
                print(f"{agent_action=}")
                print(f"{env_action=}\n")

            if env_action is not None:

                canvas = pygame.transform.flip(canvas, flip_x=False, flip_y=True)
                pygame.draw.line(
                    canvas,
                    (0, 255, 0),
                    self.env.agent.position * self.env.window_scale,
                    (self.env.agent.position + env_action * 20) * self.env.window_scale,
                    3,
                )
                canvas = pygame.transform.flip(canvas, flip_x=False, flip_y=True)

            screen.blit(canvas, (0, 0))

            pygame.display.update()

            clock.tick(30)

        pygame.quit()

    def _event_to_world(self, event_pos: tuple[int, int]) -> np.ndarray:
        p = np.array(event_pos) / float(self.env.window_scale)
        p[1] = self.env.blueprint.world_size[1] - p[1]
        return p


if __name__ == "__main__":
    model, env, window_scale = parse_model_env_window_scale()

    Dragger(env, window_scale).run(model)
