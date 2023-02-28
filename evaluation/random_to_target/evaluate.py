import sys
from typing import Callable

import gymnasium as gym

sys.modules["gym"] = gym

from multiprocessing import Pool

import numpy as np
import pandas as pd
from stable_baselines3 import PPO


def _evaluate(model_path: str, create_env: Callable[[], gym.Env], num_episodes: int):
    model = PPO.load(model_path)
    env = create_env()

    data = dict(
        length=[],
        success=[],
        reward=[],
        start_distance=[],
        mean_velocity=[],
        mean_acceleration=[],
        min_velocity=[],
        min_acceleration=[],
        max_velocity=[],
        max_acceleration=[],
        std_velocity=[],
        std_acceleration=[],
        mean_angle_change=[],
        min_angle_change=[],
        max_angle_change=[],
        std_angle_change=[],
    )

    current_episode_length = 0
    current_episode_reward = 0

    observation, info = env.reset()

    current_start_distance = np.linalg.norm(env.target - env.agent.position)

    current_velocities = [env.agent.velocity.copy()]
    current_angles = [env.agent.angle]

    while len(data["length"]) < num_episodes:
        action, _ = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action)

        current_episode_length += 1
        current_episode_reward += reward

        current_velocities.append(env.agent.velocity)
        current_angles.append(env.agent.angle)

        if terminated or truncated:
            data["success"].append(info.get("is_success", False))

            observation, info = env.reset()

            data["length"].append(current_episode_length)
            data["reward"].append(current_episode_reward)
            data["start_distance"].append(current_start_distance)

            velocities = np.array(current_velocities)
            accelerations = velocities[1:] - velocities[:-1]

            v = np.linalg.norm(velocities[1:], axis=1)
            a = np.linalg.norm(accelerations, axis=1)

            data["mean_velocity"].append(v.mean())
            data["mean_acceleration"].append(a.mean())
            data["min_velocity"].append(v.min())
            data["min_acceleration"].append(a.min())
            data["max_velocity"].append(v.max())
            data["max_acceleration"].append(a.max())
            data["std_velocity"].append(v.std())
            data["std_acceleration"].append(a.std())

            angles = np.array(current_angles)
            angle_changes = np.abs(np.diff(angles))
            data["mean_angle_change"].append(angle_changes.mean())
            data["min_angle_change"].append(angle_changes.min())
            data["max_angle_change"].append(angle_changes.max())
            data["std_angle_change"].append(angle_changes.std())

            current_episode_length = 0
            current_episode_reward = 0

            current_start_distance = np.linalg.norm(env.target - env.agent.position)
            current_velocities = [env.agent.velocity.copy()]
            current_angles = [env.agent.angle]

    df = pd.DataFrame(data)

    return df


def evaluate(
    model_path: str,
    create_env: Callable[[], gym.Env],
    num_episodes: int,
    num_processes: int,
):
    with Pool(num_processes) as pool:
        df = pd.concat(
            pool.starmap(
                _evaluate,
                [(model_path, create_env, num_episodes // num_processes)]
                * num_processes,
                chunksize=1,
            )
        )
    return df
