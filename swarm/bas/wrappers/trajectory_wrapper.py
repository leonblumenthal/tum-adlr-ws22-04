import gymnasium as gym


class TrajectoryWrapper(gym.Wrapper):
    """BAS wrapper to accumulate the agent's trajectory and store it inside info if the episode ends."""

    def __init__(self, env: gym.Env):
        """Initialize the wrapper.

        Args:
            env: (Wrapped) BAS environment.
        """
        super().__init__(env)

    def reset(self, **kwargs):
        """Store first position of the agent's trajectory."""
        observation, info = self.env.reset(**kwargs)

        self._trajectory = [self.agent.position]

        return observation, info

    def step(self, action):
        """Accumulate the agent's trajectory and store it in info if the episode ends."""
        observation, reward, terminated, truncated, info = self.env.step(action)

        self._trajectory.append(self.agent.position)

        if terminated or truncated:
            info["agent_trajectory"] = self._trajectory

        return observation, reward, terminated, truncated, info
