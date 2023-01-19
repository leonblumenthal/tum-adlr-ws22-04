"""Collection of (in)complete environment used for experimentation."""
import gymnasium as gym
import numpy as np

from swarm.bas import Agent, BASEnv, Blueprint, RenderWrapper, Swarm, wrappers


class SneakyGridAndGoalEnv(gym.Wrapper):
    def __init__(self, window_scale: float = 8):
        blueprint = Blueprint(world_size=np.array([200, 100]))
        agent = Agent(
            radius=1,
            max_velocity=1,
            max_acceleration=0.2,
            reset_position=np.array([5, 50]),
        )
        swarm = Swarm(
            num_boids=100,
            radius=2,
            max_velocity=None,
            reset_positions=np.stack(
                np.meshgrid(np.linspace(55, 145, 10), np.linspace(5, 95, 10)), -1
            ).reshape(-1, 2),
            max_acceleration=0,
            separation_range=0,
            cohesion_range=0,
            alignment_range=0,
            steering_weights=0,
            obstacle_margin=0,
        )
        env = BASEnv(blueprint, agent, swarm)

        env = wrappers.SectionObservationWrapper(
            env, num_sections=16, max_range=20, subtract_radius=False
        )
        env = wrappers.TargetRewardWrapper(
            env, position=np.array([195, 50]), distance_reward_scale=1, target_reward=10
        )
        env = wrappers.BoidCollisionWrapper(
            env, collision_termination=True, collision_reward=-10
        )

        env = RenderWrapper(env, window_scale=window_scale)
        env = wrappers.FlattenObservationWrapper(env)

        env = gym.wrappers.TimeLimit(env, 500)

        super().__init__(env)


class GoalInsideGridEnv(gym.Wrapper):
    def __init__(
        self,
        collision_termination: bool = False,
        collision_reward: int = 0,
        window_scale: float = 5,
    ):
        blueprint = Blueprint(
            world_size=np.array([200, 200]),
        )
        agent = Agent(
            radius=1,
            max_velocity=1,
            max_acceleration=0.2,
        )
        swarm = Swarm(
            num_boids=100,
            radius=2,
            max_velocity=None,
            reset_positions=np.stack(
                np.meshgrid(np.linspace(55, 145, 10), np.linspace(55, 145, 10)), -1
            ).reshape(-1, 2),
            max_acceleration=0.1,
            separation_range=10,
            cohesion_range=20,
            alignment_range=20,
            steering_weights=(1.1, 1, 1),
            obstacle_margin=5,
        )
        env = BASEnv(blueprint, agent, swarm)

        target = np.array([100, 100])
        env = wrappers.TargetDirectionAndSectionObservationWrapper(
            env, num_sections=8, max_range=20, position=target, subtract_radius=True
        )
        env = wrappers.TargetRewardWrapper(
            env,
            position=target,
            distance_reward_scale=1,
            target_radius=3,
            target_reward=100,
        )
        env = wrappers.BoidCollisionWrapper(
            env,
            collision_termination=collision_termination,
            collision_reward=collision_reward,
            add_reward=True,
        )

        env = RenderWrapper(env, window_scale=window_scale)
        env = wrappers.FlattenObservationWrapper(env)

        env = gym.wrappers.TimeLimit(env, 500)

        super().__init__(env)


class FollowBoidsEnv(gym.Wrapper):
    def __init__(self):
        blueprint = Blueprint(
            world_size=np.array([100, 100]),
        )
        agent = Agent(
            radius=1,
            max_velocity=1,
            max_acceleration=0.2,
            reset_position=np.array([5, 50]),
        )
        swarm = (
            Swarm(
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

        env = BASEnv(blueprint, agent, swarm)
        env = wrappers.NumNeighborsRewardWrapper(env, max_range=20)
        env = wrappers.SectionAndVelocityObservationWrapper(
            env, num_sections=8, max_range=20
        )
        env = wrappers.FlattenObservationWrapper(env)
        env = gym.wrappers.TimeLimit(env, 500)
        super().__init__(env)
