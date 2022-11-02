import gymnasium as gym
from gymnasium.envs.registration import register
from gymnasium.utils.play import play


register(
    id="SwarmEnv",
    entry_point="swarm_env:SwarmEnv",
)

env = gym.make("SwarmEnv", render_mode="rgb_array")

# observation, info = env.reset(seed=42)
# for _ in range(1000):
#     action = env.action_space.sample()
#     observation, reward, terminated, truncated, info = env.step(action)

#     if terminated or truncated:
#         observation, info = env.reset()
# env.close()

play(
    env,
    keys_to_action={
        "z": 0,
        "d": 1,
        "w": 2,
        "a": 3,
        "s": 4,
    },
    seed=187
)
