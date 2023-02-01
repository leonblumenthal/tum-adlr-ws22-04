import sys
from pathlib import Path
from typing import Callable

import gymnasium as gym

sys.modules["gym"] = gym

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from swarm.training.callbacks import DrawTrajectoriesCallback, SuccessRateCallback


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
                tensorboard_log=runs_path / experiment_path / "tensorboard_train",
            )
        else:
            model.set_env(env)

        model.learn(
            total_timesteps=num_steps,
            callback=[SuccessRateCallback(), DrawTrajectoriesCallback(create_env())],
            reset_num_timesteps=False,
        )

        model.save(runs_path / experiment_path / f"model_{chapter}")
