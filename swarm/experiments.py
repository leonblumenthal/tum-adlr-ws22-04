"""Collection of (in)complete environment used for experimentation."""
import gymnasium as gym
import numpy as np

from swarm.bas import Agent, BASEnv, Blueprint, RenderWrapper, Swarm, wrappers


# TODO: WIP
class GridAndGoalEnv(gym.Wrapper):
    def __init__(self, window_scale: float = 8):
        blueprint = Blueprint(
            world_size=np.array([200, 100]),
        )
        agent = Agent(
            radius=1,
            max_velocity=1,
            max_acceleration=0.2,
            reset_position=np.array([5, 95]),
        )
        swarm = Swarm(
            num_boids=100,
            radius=2,
            max_velocity=None,
            reset_positions=np.stack(
                np.meshgrid(np.linspace(55, 145, 10), np.linspace(1, 99, 10)), -1
            ).reshape(-1, 2),
            max_acceleration=0,
            separation_range=0,
            cohesion_range=0,
            alignment_range=0,
            steering_weights=0,
            obstacle_margin=0,
        )
        env = BASEnv(blueprint, agent, swarm)

        target = np.array([195, 5])
        env = wrappers.TargetDirectionAndSectionObservationWrapper(
            env, num_sections=16, max_range=20, position=target, subtract_radius=True
        )
        env = wrappers.DistanceToTargetRewardWrapper(env, position=target)
        env = wrappers.BoidCollisionWrapper(
            env, collision_termination=False, collision_reward=-50, add_reward=False
        )
        env = RenderWrapper(env, window_scale=window_scale)
        # env = wrappers.RelativeRotationWrapper(env)
        env = wrappers.FlattenObservationWrapper(env)
        env = gym.wrappers.TimeLimit(env, 2000)

        super().__init__(env)
