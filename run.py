"""Play the BAS environment with some wrappers, especially rendering."""

import gymnasium as gym
import numpy as np
import torch

from swarm.bas import Agent, BASEnv, Blueprint, Swarm, wrappers, RenderWrapper
from ppo_continuous import ActorCriticNetwork


# Create an example environment with some wrappers.
env = BASEnv(
    blueprint=Blueprint(
        world_size=np.array([100, 100]),
    ),
    agent=Agent(
        radius=1,
        max_velocity=1.2,
        max_acceleration=0.1,
        reset_position=np.array([50, 50]),
    ),
    swarm=Swarm(
        num_boids=100,
        radius=1,
        max_velocity=1,
        max_acceleration=0.1,
        separation_range=5,
        cohesion_range=10,
        alignment_range=10,
        steering_weights=(1.1, 1, 1),
        obstacle_margin=3,
    ),
)

env = wrappers.NumNeighborsRewardWrapper(env, max_range=20)
env = wrappers.SectionAndVelocityObservationWrapper(env, num_sections=8, max_range=20)
env = RenderWrapper(env)
env = wrappers.FlattenObservationWrapper(env)
env = gym.wrappers.HumanRendering(env)

observation, info = env.reset(seed=42)

actor_critic: ActorCriticNetwork = torch.load("actor_critic")
actor_critic.to("cpu")

for _ in range(10000):
    observation = torch.Tensor(observation).unsqueeze(0)
    action, _, _, _ = actor_critic.get_action_and_value(observation)

    observation, reward, terminated, truncated, info = env.step(action[0].numpy())

    if terminated or truncated:
        observation, info = env.reset()

env.close()
