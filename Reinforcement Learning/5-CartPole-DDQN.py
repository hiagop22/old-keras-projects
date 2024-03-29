#!usr/bin/env python3
#-*- coding: utf-8 -*-
"""
    Created on 05-05-2020 / 11:39:05
    @author: Hiago dos Santos
    @e-mail: hiagop22@gmail.com
"""

# %%
import gym
import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import keras.backend as k

import tensorflow as tf

# %%
MAX_EPISODES = 1000

class DDQNetork(object):
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.max_epsilon = 1.0
        self.min_epsilon = 0.01
        self.decay_epsilon = 0.999
        self.learning_rate = 0.01
        self.model = self.built_model()
        self.target_model = self.built_model()
        self.update_target_model()

    def _huber_loss(self, y_true, y_pred, clip_delta=1.0):
        error = y_true - y_pred
        cond = k.abs(error) <= clip_delta

        square_loss = 0.5*k.square(error)
        quadratic_loss = 0.5*k.square(clip_delta) + clip_delta*(k.abs(error) - clip_delta)

        return k.mean(tf.where(cond, square_loss, quadratic_loss))
    
    def built_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss=self._huber_loss, optimizer=Adam(lr=self.learning_rate))

        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                t = self.target_model.predict(next_state)[0]
                target[0][action] = reward + self.gamma*np.amax(t)
            self.model.fit(state, target, epochs=1, verbose=0)

    def update_epsilon(self):
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.decay_epsilon

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


# %%
env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = DDQNetork(state_size, action_size)
# agent.load("./save/cartpole-ddqn.h5")
done = False
batch_size = 32

for episode in range(MAX_EPISODES):
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    for time in range(500):
        # env.render()
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        
        next_state = np.reshape(next_state, [1, state_size])
        agent.memorize(state, action, reward, next_state, done)
        state = next_state
        if done:
            agent.update_target_model()
            print("episode: {}/{}, score: {}, e: {:.2}"
                    .format(episode, MAX_EPISODES, time, agent.epsilon))
            break
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
    agent.update_epsilon()
