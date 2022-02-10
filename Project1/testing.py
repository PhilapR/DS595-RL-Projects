#%%
import numpy as np
import gym
env = gym.make("FrozenLake-v1")
env = env.unwrapped
# %%
env.step(1)
env.render()
# %%
