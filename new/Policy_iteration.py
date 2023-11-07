#%%
import gymnasium as gym
import numpy as np
env=gym.make("FrozenLake-v1",render_mode='human')
env.reset()
env.render()
#%%
def value_function(policy):
    num_iterations=1000
    gamma=1.0
    threshold=1e-20
    value_table=np.zeros(env.observation_space.n)
    for i in range(num_iterations):
        updated_table=np.copy(value_table)
        for s in range(env.observation_space.n):
            a=policy[s]
            value_table[s]=sum([prob*(r+gamma*updated_table[s_])
                           for prob,s_,r,_ in env.P[s][a]])
        if np.sum(np.fabs(updated_table-value_table))<=threshold:
            break
    return value_table
#%%
def extract_policy(value_table):
    gamma=1.0
    policy=np.zeros(env.observation_space.n)
    for s in range(env.observation_space.n):
        Q_values=[sum([prob*(r+gamma*value_table[s_])
                       for prob,s_,r,_ in env.P[s][a]])
                          for a in range(env.action_space.n)]
        policy[s]=np.argmax(np.array(Q_values))
    return policy
#%%
def policy_iteration(env):
    num_iterations=1000
    policy=np.random.randint(env.action_space.n,size=env.observation_space.n)
    for i in range(num_iterations):
        optimal_value_function=value_function(policy)
        new_policy=extract_policy(optimal_value_function)
        if np.all(policy==new_policy):
            break
        policy=new_policy
    return policy
#%%
optimal_policy=policy_iteration(env)
print(optimal_policy)