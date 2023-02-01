"""Collection of (in)complete environment used for experimentation."""
from typing import Callable

import gymnasium as gym
import numpy as np

from swarm.bas import (
    Agent,
    BASEnv,
    Blueprint,
    RenderWrapper,
    Swarm,
    wrappers,
    InstantSpawner,
    BernoulliSpawner,
    SwarmConfig
)


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
            SwarmConfig(
                num_boids=100,
                radius=2,
                max_velocity=None,
                max_acceleration=0,
                separation_range=0,
                cohesion_range=0,
                alignment_range=0,
                steering_weights=0,
                obstacle_margin=0,
            ),
            InstantSpawner(
                spawn_positions=np.stack(
                np.meshgrid(np.linspace(55, 145, 10), np.linspace(5, 95, 10)), -1
            ).reshape(-1, 2),
            )
        )
        env = BASEnv(blueprint, agent, swarm)

        env = wrappers.SectionObservationWrapper(
            env, num_sections=16, max_range=20, subtract_radius=False
        )
        env = wrappers.TargetRewardWrapper(
            env, position=np.array([195, 50]), target_reward=10
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
        window_scale: float | None = 5,
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
             SwarmConfig(
                num_boids=100,
                radius=2,
                max_velocity=None,
                max_acceleration=0.1,
                separation_range=10,
                cohesion_range=20,
                alignment_range=20,
                steering_weights=(1.1, 1, 1, 0),
                obstacle_margin=5,
            ),
            InstantSpawner(
                spawn_positions=np.stack(
                np.meshgrid(np.linspace(55, 145, 10), np.linspace(55, 145, 10)), -1
            ).reshape(-1, 2),
            )
        )

        env = BASEnv(blueprint, agent, swarm)

        target = np.array([100, 100])
        env = wrappers.TargetDirectionAndSectionObservationWrapper(
            env, num_sections=8, max_range=20, position=target, subtract_radius=True
        )
        env = wrappers.TargetRewardWrapper(
            env,
            position=target,
            target_radius=3,
            target_reward=1,
        )
        env = wrappers.BoidCollisionWrapper(
            env,
            collision_termination=collision_termination,
            collision_reward=collision_reward,
            add_reward=True,
        )

        if window_scale is not None:
            env = RenderWrapper(env, window_scale=window_scale)

        env = wrappers.FlattenObservationWrapper(env)

        env = gym.wrappers.TimeLimit(env, 500)

        super().__init__(env)


class FollowBoidsEnv(gym.Wrapper):
    def __init__(self,
        collision_termination: bool = False,
        collision_reward: int = 0,
        window_scale: float = 5,
    ):
        blueprint = Blueprint(
            world_size=np.array([100, 100]),
        )
        agent = Agent(
            radius=1,
            max_velocity=1,
            max_acceleration=0.2,
            reset_position=np.array([5, 50]),
        )
        swarm = Swarm(
            SwarmConfig(
                num_boids=100,
                radius=1,
                max_velocity=1,
                max_acceleration=0.1,
                separation_range=5,
                cohesion_range=10,
                alignment_range=10,
                steering_weights=(1.1, 1, 1, 0),
                obstacle_margin=3,
            ),
            InstantSpawner()
        )

        env = BASEnv(blueprint, agent, swarm)
        env = wrappers.NumNeighborsRewardWrapper(env, max_range=20)
        env = wrappers.SectionAndVelocityObservationWrapper(
            env, num_sections=8, max_range=20
        )

        env = RenderWrapper(env, window_scale=window_scale)
        env = wrappers.FlattenObservationWrapper(env)
        env = gym.wrappers.TimeLimit(env, 500)
        super().__init__(env)


class SourceSinkEnv(gym.Wrapper):
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
            max_velocity=2,
            max_acceleration=0.2,
        )
        swarm = Swarm(
            SwarmConfig(
               num_boids=100,
                radius=2,
                max_velocity=1.5,
                max_acceleration=0.1,
                separation_range=10,
                cohesion_range=20,
                alignment_range=20,
                steering_weights=(2, 1, 1, 0.8),
                obstacle_margin=5,
                target_position=agent.position,
                target_despawn=False,
            ),
            BernoulliSpawner(spawn_probability=0.08, spawn_radius=30,spawn_position=np.array([150, 150]))
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

class RelativeDynamicRandomTargetEnv(gym.Wrapper):
    """Environment with a random target and a dynamic swarm."""

    def __init__(
        self,
        swarm_warmup_steps: int | tuple[int, int] = 0,
        num_sections: int = 8,
        collision_termination: bool = False,
        collision_reward: int = 0,
        add_collision_reward: bool = True,
        distance_reward_transform: Callable[[float], float] = lambda d: -d,
        target_radius: float = 3,
        target_reward: float = 3,
        time_limit: int = 1000,
        window_scale: float | None = 5,
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
            SwarmConfig(
               num_boids=100,
                radius=2,
                max_velocity=1,
                max_acceleration=0.1,
                separation_range=10,
                cohesion_range=20,
                alignment_range=20,
                steering_weights=(1.1, 1, 1, 0.0),
                obstacle_margin=3,
            ),
            InstantSpawner()
        )

        env = BASEnv(blueprint, agent, swarm)

        env = wrappers.SwarmWarmUpWrapper(env, swarm_warmup_steps)

        env = wrappers.RandomTargetWrapper(env)
        target = env.target

        env = wrappers.TargetDirectionAndSectionObservationWrapper(
            env, num_sections=num_sections, max_range=20, position=target, subtract_radius=True
        )
        env = wrappers.TargetRewardWrapper(
            env,
            position=target,
            target_radius=target_radius,
            target_reward=target_reward,
            distance_reward_transform=distance_reward_transform,
        )
        env = wrappers.BoidCollisionWrapper(
            env,
            collision_termination=collision_termination,
            collision_reward=collision_reward,
            add_reward=add_collision_reward,
        )

        if window_scale is not None:
            env = RenderWrapper(env, window_scale=window_scale)

        env = wrappers.RelativeRotationWrapper(env)
        env = wrappers.FlattenObservationWrapper(env)

        env = gym.wrappers.TimeLimit(env, time_limit)

        env = wrappers.TrajectoryWrapper(env)

        super().__init__(env)