import time
from typing import Any
from tqdm import tqdm

import gymnasium as gym
import numpy as np
from gymnasium.envs.registration import register
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pygame

from swarm.swarm_and_agent_env import SwarmAndAgentEnvConfig

register(
    id="SwarmAndAgentEnv",
    entry_point="swarm.swarm_and_agent_env:SwarmAndAgentEnv",
)



dqn = torch.load('dqn1')

def get_action(observation) -> np.ndarray:
    with torch.no_grad():
        out = dqn(torch.from_numpy(observation).flatten().unsqueeze(0).float())
    action_i = torch.argmax(out)
    angle = action_i * 2 * np.pi / 8
    action = np.array([np.cos(angle), np.sin(angle)])
    return action


env = gym.make("SwarmAndAgentEnv", render_mode='human')
observation, info = env.reset()

for _ in range(10000):
    observation, reward, _, _, info = env.step(get_action(observation))
    print(reward, info)
    if pygame.key.get_pressed()[pygame.K_q]:
        break

