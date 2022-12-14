"""TODO: WIP"""
import matplotlib.pyplot as plt
import numpy as np
import torch

from ppo_continuous import ActorCriticNetwork
from swarm.experiments import GridAndGoalEnv

env = GridAndGoalEnv(window_scale=8)
env.reset()
actor_critic: ActorCriticNetwork = torch.load("actor_critic")

width, height = env.blueprint.world_size.astype(int)
num_width, num_height = width // 2, height // 2

xs, ys = np.meshgrid(
    np.linspace(0, width, num_width), np.linspace(0, height, num_height)
)
xs = xs.flatten()
ys = ys.flatten()

observations = np.zeros((num_width * num_height, *env.observation_space.shape))

for i, (x, y) in enumerate(zip(xs, ys)):
    env.agent.position = np.array([x, y])
    env.step(np.zeros((2,), dtype=float))
    observations[i] = env.cached_observation.flatten()

n = 100
actions, _, _, _ = actor_critic.get_action_and_value(
    torch.from_numpy(observations).float()
)
for _ in range(n - 1):
    new_actions, _, _, _ = actor_critic.get_action_and_value(
        torch.from_numpy(observations).float()
    )
    actions += new_actions

us, vs = actions.T / n


fig, ax = plt.subplots(figsize=(15, 8))
ax.imshow(env.render())

ax.quiver(xs * env.window_scale, (height - ys) * env.window_scale, us, vs)

fig.show()
plt.show()
