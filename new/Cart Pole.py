# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 23:52:48 2023

@author: madha
"""

import gymnasium as gym
import pandas as pd
env=gym.make("CartPole-v1",render_mode="human")
env.reset()
env.render()
print(env.observation_space)
print(env.action_space)
d={}
for e in range(1,21):
    t_r=0
    env.reset()
    for t in range(1,21):
        r=env.action_space.sample()
        sp,reward,done,info,tp=env.step(r)
        env.render()
        if(reward == 1):
            t_r=t_r+reward
        if done:
            break
    if(e%5 == 0):
        d[e]=t_r
        #%%
dt=pd.DataFrame(d.items(),columns=["Episode","Return"])
print(dt)