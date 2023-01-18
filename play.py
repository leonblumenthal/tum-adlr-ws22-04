"""Play the BAS environment with some wrappers, especially rendering."""

from gymnasium.utils.play import play
import numpy as np

from swarm.bas import Agent, BASEnv, Blueprint, Swarm, wrappers, RenderWrapper


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
env = wrappers.DiscreteActionWrapper(env, num_actions=5)
env = wrappers.SectionObservationWrapper(env, num_sections=8, max_range=20)
env = RenderWrapper(env)
env = wrappers.FlattenObservationWrapper(env)

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
