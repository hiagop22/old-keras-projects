# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import gym
import random


# %%
env = gym.make("Taxi-v3")
env.render()


# %%
action_size = env.action_space.n
print("Action size: " + str(action_size))
state_size = env.observation_space.n
print("State size: " + str(state_size))


# %%
qtable = np.zeros((state_size, action_size))
print(qtable)


# %%
total_episodes = 50000
total_test_episodes = 10000
max_steps = 99

learning_rate = 0.7
gamma = 0.618

# Exploration parameters
epsilon = 1
max_epsilon = 1
min_epsilon = 0.01
decay_rate = 0.01 # Exponential decay rate


# %%
for episode in range(total_episodes):
    state = env.reset() # reset the environment
    step = 0
    
    done = False # if the taxi driver dropp the passanger at the point
    
    for step in range(max_steps):
        # Choose a value between 0 and 1, if the value > epsilon it implies that we'll choose exploitation. 
        # Something else, we uso exploration
        exp_exp_tradeoff = random.uniform(0,1)
        
        if exp_exp_tradeoff > epsilon:
            action = np.argmax(qtable[state, :])
        
        # else doing a random choise --> exploration
        else:
            action = env.action_space.sample()
            
        new_state, reward, done, info = env.step(action)
        
        # Update Q(s,a) := Q(s,a) + lr(r + gamma*argmax(Q(s',a')) - Q(s,a))
        qtable[state, action] += learning_rate*(reward + gamma*(np.max(qtable[new_state, :])) - qtable[state, action]) 
        
        state = new_state
        
        if done:
            break
            
    epsilo = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*epsilon)

# %%
print(qtable)

# %%
env.reset()
rewards = []

for episode in range(total_test_episodes):
    done = False
    state = env.reset()
    step = 0
    total_rewards = 0
    
    for step in range(max_steps):
        action = np.argmax(qtable[state,:])
        
        new_state, reward, done, info = env.step(action)
        total_rewards += reward
        
        if done:
            rewards.append(total_rewards)
#             print("Score ", str(total_rewards))
            break
            
        state = new_state

env.close()
print("Score over time ", str(sum(rewards) / total_test_episodes) )

