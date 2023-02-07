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
):
    last = None
    while env is not None:
        if isinstance(env, wrapper_type):
            new_wrapper = new_wrapper_type(env.env, **new_wrapper_kwargs)
            last.env = new_wrapper
            return
        last = env
        env = env.env
