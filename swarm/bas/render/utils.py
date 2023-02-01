import gymnasium as gym

from swarm.bas.render.wrapper import RenderWrapper
from swarm.bas.wrappers import FlattenObservationWrapper


def inject_render_wrapper(env: gym.Wrapper, **render_wrapper_kwargs):
    """Inject `RenderWrapper` into the env/wrapper chain below the `FlattenObservationWrapper`."""
    current_env = env
    while current_env:
        if isinstance(current_env, FlattenObservationWrapper):
            render_wrapper = RenderWrapper(current_env.env, **render_wrapper_kwargs)
            current_env.env = render_wrapper
            break
        current_env = current_env.env
