import sys

import gymnasium as gym
from swarm.bas.wrappers import TrajectoryWrapper

from swarm.bas.wrappers.utils import has_wrapper

sys.modules["gym"] = gym

import shutil
import time
from pathlib import Path

import cv2
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Image
from stable_baselines3.common.utils import safe_mean

from swarm.analysis.reward import create_reward_heatmap
from swarm.analysis.trajectories import draw_trajectories
from swarm.bas import BASEnv
from swarm.bas.render.utils import inject_render_wrapper


class DrawTrajectoriesCallback(BaseCallback):
    """Callback to create image with trajectories for each rollout.

    Important:
    A `TrajectoryWrapper` and a `Monitor` with "agent_trajectory" in `info_keywords` are required.
    """

    def __init__(self, env: BASEnv, window_scale: float = 5):
        """Initialize the callback with a renderable BAS environment."""
        super().__init__(verbose=0)

        assert has_wrapper(
            env, TrajectoryWrapper
        ), "Require TrajectoryWrapper to draw trajectories"

        env = inject_render_wrapper(env, window_scale=window_scale)
        env.reset()
        env.agent.position = env.blueprint.world_size * 2
        self.background_image = env.render()
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


class VideoRecorderCallback(BaseCallback):
    """Callback to create videos of evaluation runs in the runs directory."""

    def __init__(
        self,
        env: gym.Env,
        every_n_step: int,
        num_steps: int,
        window_scale: float,
        delete_existing_videos: bool,
    ):
        super().__init__()
        self._env = inject_render_wrapper(env, window_scale=window_scale)
        self._every_n_step = every_n_step
        self._num_steps = num_steps
        self._delete_existing_videos = delete_existing_videos


    def _on_training_start(self) -> None:
        self._video_directory = Path(self.model.tensorboard_log).parent / "videos"
        if self._delete_existing_videos:
            shutil.rmtree(self._video_directory, ignore_errors=True)
        self._video_directory.mkdir(exist_ok=True, parents=True)

    def _on_step(self) -> bool:
        if self.n_calls % (self._every_n_step // self.training_env.num_envs) == 0:
            start_time = time.perf_counter()

            env = self._env
            obs, _ = env.reset()

            sample_frame = env.render()
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_path = self._video_directory / f"{self.num_timesteps//1000:06d}k.mp4"
            video_writer = cv2.VideoWriter(
                str(video_path), fourcc, 30, sample_frame.shape[:2]
            )

            for _ in range(self._num_steps):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, _ = env.step(action)

                video_writer.write(env.render())

                if terminated or truncated:
                    obs, _ = env.reset()

            video_writer.release()

            elapsed_time = time.perf_counter() - start_time
            print(
                f"Created video of {self._num_steps} steps on step {self.num_timesteps} in {elapsed_time:.01f} seconds"
            )

        return True
