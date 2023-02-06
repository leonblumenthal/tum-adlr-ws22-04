import sys

import gymnasium as gym

sys.modules["gym"] = gym
from stable_baselines3 import PPO

from swarm.bas.render.utils import inject_render_wrapper
from utils import parse_model_env_window_scale


def run_ppo(model: PPO, env: gym.Env, window_scale: float):

    inject_render_wrapper(env, window_scale=window_scale)
    env = gym.wrappers.HumanRendering(env)

    observation, info = env.reset()
    step = 0
    while True:
        agent_action, _ = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(agent_action)
        step += 1

        env.render()

        print(f"{step=}")
        print(f"{agent_action=}")
        print(f"{observation=}")
        print(f"{reward}\n")

        if terminated or truncated:
            observation, info = env.reset()
            step = 0


if __name__ == "__main__":

    run_ppo(*parse_model_env_window_scale())
