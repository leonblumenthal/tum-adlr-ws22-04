from typing import Any, Type

import gymnasium as gym


def get_wrapper(
    env: gym.Wrapper, wrapper_type: Type[gym.Wrapper]
) -> gym.Wrapper | None:
    """Get first wrapper of specified type in the env/wrapper chain."""
    while env is not None:
        if isinstance(env, wrapper_type):
            return env
        env = getattr(env, "env", None)
    return None


def has_wrapper(env: gym.Wrapper, wrapper_type: Type[gym.Wrapper]) -> bool:
    """Check whether wrapper of specified type is in the env/wrapper chain."""
    return get_wrapper(env, wrapper_type) is not None


def replace_wrapper(
    env: gym.Wrapper,
    wrapper_type: Type[gym.Wrapper],
    new_wrapper_type: Type[gym.Wrapper],
    new_wrapper_kwargs: dict[str, Any] = {},
) -> gym.Wrapper:
    """Replace first existence of specified wrapper type with a new wrapper. If not present the replacement is ignored."""
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
        env = getattr(env, "env", None)
    return first
