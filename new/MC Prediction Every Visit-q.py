import numpy as np
import gymnasium as gym
import pandas as pd
from collections import defaultdict
env=gym.make("Blackjack-v1",render_mode="human")
env.reset()
env.render()
#%%
def policy(state):
    return 0 if state[0]>19 else 1
#%%
def generate_episode(policy):
    num_iterations=20
    episode=[]
    state=env.reset()
    state=state[0]
    for i in range(num_iterations):
        action=policy(state)
        ns,reward,done,info,prob=env.step(action)
        env.render()
        episode.append((state,reward,action))
        if done:
            break
        state=ns
    return episode
#%%
total_return=defaultdict(float)
N=defaultdict(float)
num_iterations=10
for i in range(num_iterations):
    episodes=generate_episode(policy)
    states,rewards,actions=zip(*episodes)
    for t,(state,action) in enumerate(zip(states,actions)):
    
            R=sum(rewards[t:])
            total_return[(state,action)]+=R
            N[(state,action)]+=1
#%%
Total_Return=pd.DataFrame(total_return.items(),columns=["(State,Action)","Total_Return"])
N=pd.DataFrame(N.items(),columns=["(State,Action)","N"])
DF=pd.merge(Total_Return,N,on="(State,Action)")
DF["Value"]=DF["Total_Return"]/DF["N"]
print(DF.head(10))