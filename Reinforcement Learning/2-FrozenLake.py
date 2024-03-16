#!usr/bin/env python3
#-*- coding: utf-8 -*-
"""
    Created on 04-20-2020 / 12:57:10
    @author: Hiago dos Santos
    @e-mail: hiagop22@gmail.com
"""
# Observation: This game doesn't produce a expected qtable

#%%
import numpy as np
import gym
import random

# %%
env = gym.make("FrozenLake-v0", is_slippery=False)
env.render()

# %%
action_size = env.action_space.n
print("Action_size: ", action_size)
state_size = env.observation_space.n
print("State_size: ", state_size )

# %%
qtable = np.zeros((state_size, action_size))
print(qtable)

# %%
# Hyperparameters
total_episodes = 15000
max_steps = 99
learning_rate = 0.6
gamma = 0.8

# Exploration parameters
epsilon = 1
max_epsilon = 1
min_epsilon = 0.01
decay_rate = 0.01

# %%
epsilon = 1
rewards = []
for episode in range(total_episodes):
    state = env.reset()
    done = False
    total_rewards = 0

    for step in range(max_steps):
        exp_exp_tradeoff = random.uniform(0,1)

        if exp_exp_tradeoff > epsilon:
            action = np.argmax(qtable[state, :])
        else:
            action = env.action_space.sample()
        
        new_state, reward, done, info = env.step(action)
        qtable[state, action] = qtable[state, action] + \
        learning_rate * (reward + gamma * np.max(qtable[new_state, :]) \
        - qtable[state, action])

        total_rewards += reward

        state = new_state

        if done:
            break
    # if episode % 5:
    #     print(qtable)
    epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-episode*decay_rate)
    rewards.append(total_rewards)

# %%
print(qtable)
state = env.reset()
env.render()
for x in range(max_steps):
    
    action = np.argmax(qtable[state, :])
    new_state, reward, done, info = env.step(action)
    print("State: %d Action: %d" %(state, action))
    state = new_state
    env.render()
    if done:
        break

env.close()