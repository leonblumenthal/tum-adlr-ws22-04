import gymnasium as gym

from bas.render.wrapper import RenderWrapper
from bas.wrappers import FlattenObservationWrapper


def inject_render_wrapper(env: gym.Wrapper, **render_wrapper_kwargs) -> gym.Wrapper:
    """Moves the `FlattenObservationWrapper` to the outermost position of all wrappers if it is present and wraps the `RenderWrapper` around the env but inside the `FlattenObservationWrapper` to ensure correct rendering."""
    current_env = env
    last_seen = None
    first_env = env
    while current_env:
        if isinstance(current_env, FlattenObservationWrapper):
            # skip original FlattenObservationWrapper
            if last_seen is None:
                first_env = current_env.env
            else:
                last_seen.env = current_env.env
            return FlattenObservationWrapper(
                RenderWrapper(first_env, **render_wrapper_kwargs)
            )
        last_seen = current_env
        current_env = getattr(current_env, "env", None)
    return RenderWrapper(first_env, **render_wrapper_kwargs)
