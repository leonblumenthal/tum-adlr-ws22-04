"""Play the BAS environment with some wrappers, especially rendering."""

from gymnasium.utils.play import play
import numpy as np

from swarm.experiments import GoalInsideGridEnv
from swarm.bas import wrappers


# Create an example environment with some wrappers.
env = GoalInsideGridEnv()
env = wrappers.DiscreteActionWrapper(env, 5)
play(
    env,
    keys_to_action=dict(
        z=0,  # no movement
        d=1,  # right
        w=2,  # up
        a=3,  # left
        s=4,  # down
    ),
)
