import gymnasium as gym
import pandas as pd
env=gym.make("FrozenLake-v1",render_mode='human')
env.reset()
env.render()
d={}
#%%
print(env.observation_space)
print(env.action_space)
#%%
print(env.P[0][2])
#%%
total_return=0
for e in range(1,31):
    env.reset()
    total_return=0
    for t in range(1,31):
        rand=env.action_space.sample()
        ns,reward,done,info,tp=env.step(rand)
        env.render()
        if done:
            if reward==1:
                total_return=1
            break
    if e%3==0:
        d[e]=total_return
#%%
df=pd.DataFrame(d.items(),columns=["Episode","Return"])
print(df)
                