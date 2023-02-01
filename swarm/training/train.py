import sys
from typing import Callable
from pathlib import Path

import gymnasium as gym

sys.modules["gym"] = gym

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Image
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import safe_mean
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3 import PPO


from swarm.analysis.reward import create_reward_heatmap
from swarm.analysis.trajectories import draw_trajectories
from swarm.bas import BASEnv
from swarm.bas.render.wrapper import RenderWrapper


class DrawTrajectoriesCallback(BaseCallback):
    """Callback to create image with trajectories for each rollout.

    Important:
    A `TrajectoryWrapper` and a `Monitor` with "agent_trajectory" in `info_keywords` are required.
    """

    def __init__(self, env: BASEnv, window_scale: float = 5):
        """Initialize the callback with a renderable BAS environment."""
        super().__init__(verbose=0)

        width, height = env.blueprint.world_size.astype(int)
        self.background_image = (
            np.ones((height * window_scale, width * window_scale, 3)).astype(np.uint8)
            * 255
        )
        self.window_scale = window_scale

    def _on_rollout_end(self):
        """Draw the trajectories from the last 100 episodes and log them."""
        trajectories = [
            np.array(ep_info["agent_trajectory"]) * self.window_scale
            for ep_info in self.model.ep_info_buffer
        ]
        trajectories_image = self.background_image.copy()
        draw_trajectories(
            trajectories_image,
            trajectories,
            1,
        )
        self.logger.record(
            "rollout/trajectories",
            Image(trajectories_image, "HWC"),
            exclude=("stdout", "log", "json", "csv"),
        )

        return super()._on_rollout_end()

    # This is required?!
    def _on_step(self) -> bool:
        return True


# TODO: The heatmap does not look correct in TB.
#       Using a figure instead of an image is probably better.
class RewardHeatmapCallback(BaseCallback):
    """Callback to create a heatmap of the reward before each training."""

    def __init__(self, env: BASEnv, resolution: tuple[int, int] = (200, 200)):
        """Initialize the callback with a BAS environment."""
        self.env = env
        self.resolution = resolution

        super().__init__(verbose=0)

    def _on_training_start(self) -> None:
        """Create a heatmap of the reward and log it."""
        self.env.reset()

        reward_heatmap = create_reward_heatmap(
            self.env, num_width=self.resolution[1], num_height=self.resolution[0]
        )
        reward_heatmap -= reward_heatmap.min()
        reward_heatmap /= reward_heatmap.max()
        self.model.logger.record(
            f"analysis/reward_heatmap",
            Image(reward_heatmap, "HW"),
            exclude=("stdout", "log", "json", "csv"),
        )

        return super()._on_training_start()

    # This is required?!
    def _on_step(self) -> bool:
        return True


# TODO: This should technically be included but may not work due to gymnasium stuff.
class SuccessRateCallback(BaseCallback):
    """Callback to log the success rate."""

    def _on_rollout_end(self):
        """Log the success rate over the last 100 episodes."""
        self.logger.record(
            "rollout/success_rate",
            safe_mean([ep_info["is_success"] for ep_info in self.model.ep_info_buffer]),
        )
        return super()._on_rollout_end()

    # This is required?!
    def _on_step(self) -> bool:
        return True


def create_parallel_env(
    create_env: Callable[[], gym.Wrapper],
    num_processes: int,
    info_keywords: list[str] = ["is_success"],
) -> SubprocVecEnv:
    def f():
        return Monitor(create_env(), info_keywords=info_keywords)

    return SubprocVecEnv([f] * num_processes)


def train(
    curriculum: list[int, Callable[[], gym.Env]],
    experiment_path: Path,
    num_processes: int,
    runs_path: Path = Path("runs"),
):
    model = None
    for chapter, (num_steps, create_env) in enumerate(curriculum):

        env = create_parallel_env(
            create_env,
            num_processes,
            ["is_success", "agent_trajectory"],
        )

        if model is None:
            model = PPO(
                "MlpPolicy",
                env,
                verbose=1,
                tensorboard_log=runs_path / experiment_path / "train_tensorboard",
            )
        else:
            model.set_env(env)

        model.learn(
            total_timesteps=num_steps,
            callback=[SuccessRateCallback(), DrawTrajectoriesCallback(create_env())],
            reset_num_timesteps=False,
        )

        model.save(runs_path / experiment_path / f"model_{chapter}")
