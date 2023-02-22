from typing import Any, Type

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


def replace_wrapper(
    env: gym.Wrapper,
    wrapper_type: Type[gym.Wrapper],
    new_wrapper_type: Type[gym.Wrapper],
    new_wrapper_kwargs: dict[str, Any] = {},
) -> gym.Wrapper:
    last_seen = None
    first = env
    while env is not None:
        if isinstance(env, wrapper_type):
            new_wrapper = new_wrapper_type(env.env, **new_wrapper_kwargs)
            if last_seen is None:
                return new_wrapper
            last_seen.env = new_wrapper
            return first
        last_seen = env
        env = env.env
    return first
