#!usr/bin/env python3
#-*- coding: utf-8 -*-
"""
    Created on 04-27-2020 / 16:21:15
    @author: Hiago dos Santos
    @e-mail: hiagop22@gmail.com
"""

# %%
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ReduceLROnPlateau, LearningRateScheduler
from keras.optimizers import Adam

from collections import deque
import numpy as np
import random

import gym

# %%
# Constants 
MAX_EPISODES = 1000
MAX_DESIRED_TIME = 500
MAX_MEMORY = 2000
BATCH_SIZE = 32 

# %%
class DQNetwork(object):
    def __init__(self, state_size, action_size):
        self.action_size = action_size
        self.state_size = state_size
        self.memory = deque(maxlen=MAX_MEMORY)
        self.epsilon = 1.0
        self.min_epsilon = 0.001
        self.max_epsilon = 1.0
        self.epsilon_decay = 0.01
        self.learning_rate = 0.6
        self.gamma = 0.93
        self.model = self.built_model()
        self.callbacks = self.callbacks()

    def built_model(self):
        model = Sequential()
        model.add(Dense(20, activation='relu', input_dim=self.state_size))
        model.add(Dense(20, activation='relu'))
        model.add(Dense(self.action_size, activation='softmax'))

        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate), metrics=['accuracy'])

        return model

    def act(self, state):
        if random.random() > self.epsilon:
            return np.argmax(self.model.predict(state)[0])
        
        return env.action_space.sample()

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, episode):
        minibatch = random.sample(self.memory, BATCH_SIZE)

        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma*np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target

            self.model.fit(state, target_f, epochs=1, verbose=0, 
                        batch_size=200)
                        # callbacks=self.callbacks,

            # if self.epsilon > self.min_epsilon:
            self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon)*np.exp(-episode*self.epsilon_decay)
    
    def update_learning_rate(self, _, lr):
        self.learning_rate = lr

        return self.learning_rate
    
    def callbacks(self):
        reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', 
                                    factor=0.1,
                                    patience=3,
                                    min_delta=1,
                                    min_lr=0.0001)

        lr_scheduler = LearningRateScheduler(self.update_learning_rate, verbose=0)

        return [reduce_lr, lr_scheduler]

# %%
env = gym.make('CartPole-v1')
agent = DQNetwork(env.observation_space.shape[0], env.action_space.n)

for episode in range(MAX_EPISODES):
    state = env.reset()
    state = np.reshape(state, [1,4])

    for time_t in range(MAX_DESIRED_TIME):
        # env.render()

        action = agent.act(state)

        next_state, reward, done, _ = env.step(action)
        reward = reward if not done else -10
        next_state = np.reshape(next_state, [1, 4])

        agent.memorize(state, action, reward, next_state, done)

        state = next_state

        if done:
            print('Episode: {}/{}, Reward: {}, Epsilon: {}, LR: {}'
            .format(episode, MAX_EPISODES, time_t, agent.epsilon, agent.learning_rate))
            break
        
        if (len(agent.memory) > BATCH_SIZE):
            agent.replay(episode)