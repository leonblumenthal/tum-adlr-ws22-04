import sys
from pathlib import Path
from typing import Callable

import gymnasium as gym

sys.modules["gym"] = gym

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from training.callbacks import (
    DrawTrajectoriesCallback,
    SuccessRateCallback,
    VideoRecorderCallback,
)


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
    video_every_n_steps: int | None = None,
    video_num_steps: int = 1000,
    video_window_scale: int = 3,
    runs_path: Path = Path("runs"),
    initial_model_path: Path | None = None,
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
                tensorboard_log=runs_path / experiment_path / "tensorboard_train",
            )
            if initial_model_path is not None:
                print(f"Loading parameters from {initial_model_path}")
                model.set_parameters(initial_model_path)
        else:
            model.set_env(env)

        callbacks = [SuccessRateCallback(), DrawTrajectoriesCallback(create_env())]

        if video_every_n_steps is not None:
            callbacks.append(
                VideoRecorderCallback(
                    create_env(),
                    video_every_n_steps,
                    video_num_steps,
                    video_window_scale,
                    chapter == 0,
                )
            )

        model.learn(
            total_timesteps=num_steps,
            callback=callbacks,
            reset_num_timesteps=False,
        )

        model.save(runs_path / experiment_path / f"model_{chapter}")
