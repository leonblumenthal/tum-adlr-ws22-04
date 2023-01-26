import sys
import pathlib

import gymnasium as gym

sys.modules["gym"] = gym

from stable_baselines3 import PPO
from swarm.training.train import (
    create_parallel_env,
    SuccessRateCallback,
    DrawTrajectoriesCallback,
)


from swarm.experiments import RelativeDynamicRandomTargetEnv

# Num steps, function to create env with window scale
train_steps = [
    (
        500000,
        lambda window_scale=None: RelativeDynamicRandomTargetEnv(
            window_scale=window_scale,
            collision_termination=False,
            collision_reward=0,
            add_reward=True,
            distance_reward_transform=lambda d: -d,
            target_radius=3,
            target_reward=3,
        ),
    ),
]


runs_directory = pathlib.Path("runs")
tensorboard_directory = runs_directory / "tensorboard"

experiment_name = "RelativeDynamicRandomTargetEnv/test_1"
experiment_path = runs_directory / experiment_name

num_processes = 16

if __name__ == "__main__":
    model = None
    for i, (num_steps, create_env) in enumerate(train_steps):

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
                tensorboard_log=tensorboard_directory / experiment_name,
            )
        else:
            model.set_env(env)

        model.learn(
            total_timesteps=num_steps,
            callback=[SuccessRateCallback(), DrawTrajectoriesCallback(create_env(5))],
            reset_num_timesteps=False,
        )

        model.save(experiment_path)
