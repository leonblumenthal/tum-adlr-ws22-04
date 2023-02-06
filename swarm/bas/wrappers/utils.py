from typing import Type

import gymnasium as gym


def get_wrapper(
    env: gym.Wrapper, wrapper_type: Type[gym.Wrapper]
) -> gym.Wrapper | None:
    """Get first wrapper of specified type in the env/wrapper chain."""
    while env is not None:
        if isinstance(env, wrapper_type):
            return env
        env = getattr(env, "env")
    return None
