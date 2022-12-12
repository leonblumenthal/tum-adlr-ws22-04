from typing import Any

import gymnasium as gym
from gymnasium import spaces

from swarm.bas.agent import Agent
from swarm.bas.blueprint import Blueprint
from swarm.bas.swarm import Swarm


class BASEnv(gym.Env):
    """Environment consisting of blueprint, agent, and swarm (BAS).

    This environment does not really implement observations, rewards, and rendering.
    Custom wrappers are designed to fulfill these parts in a modular way.

    The action space is clearly specified by the agent's capability
    to move based on a direction vector.
    """

    # Next direction for the agent.
    ActionType = Agent.ActionType
    # Additional Information.
    InfoType = dict[str]

    # Action corresponds to next velocity of the agent.
    # This may be scaled by the agent itself.
    action_space = Agent.action_space

    # No render mode is defined because that is handled by a custom wrapper.

    def __init__(self, blueprint: Blueprint, agent: Agent, swarm: Swarm):
        """Initialize the environment.

        Args:
            blueprint: Blueprint for this environment.
            agent:  Agent for this environment.
            swarm:  Swarm for this environment.
        """
        self.blueprint = blueprint
        self.agent = agent
        self.swarm = swarm

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[Any, dict]:
        """Reset the agent and swarm.

        The returned observation should be replaced by a wrapper.
        """
        super().reset(seed=seed, options=options)

        self.agent.reset(world_size=self.blueprint.world_size, np_random=self.np_random)
        self.swarm.reset(world_size=self.blueprint.world_size, np_random=self.np_random)

        # Dummy observation which is not None.
        observation = True
        info = {}

        return observation, info

    def step(self, action: ActionType) -> tuple[Any, float, bool, bool, dict]:
        """Performs step for the agent and swarm.

        The returned observation and reward should be replaced by wrappers.
        """

        self.swarm.step()
        self.agent.step(action)

        # Dummy observation which is not None.
        observation = True
        reward = None
        terminated = False
        truncated = False
        info = {}

        return observation, reward, terminated, truncated, info
