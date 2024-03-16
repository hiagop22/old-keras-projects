#!usr/bin/env python3
#-*- coding: utf-8 -*-
"""
    Created on 04-29-2020 / 19:36:16
    @author: Hiago dos Santos
    @e-mail: hiagop22@gmail.com
"""
# %%
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Dense
from keras.optimizers import Adam
from keras import backend as k

import retro

import cv2
import numpy as np
import random 
from collections import deque
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
import warnings # This ignore all warning messages that are showed during the training because skimage

warnings.filterwarnings('ignore')

# %%
k.set_image_data_format('channels_last')
k.image_data_format()

# %%
#### HyperParameters ####
STACK_FRAME_SIZE = 4    # amount of stacked frames to get moviment information from the game
BATCH_SIZE = 64         # amount of previous information restored from the memory
MEMORY_SIZE = 1000    # amount of maximum previous information stored in the memory
MAX_EPISODES = 500 
MAX_STEPS = 50000
RENDER_TEST_PLAY = False
STATE_SIZE = (84, 84, 4) # Input = (width, heigh, channels)

# %%
class DQNetork(object):
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = 1.0
        self.max_epsilon = 1.0
        self.min_epsilon = 0.001
        self.decay_epsilon = 0.005 # Hightest it to use more frequently previous experiences
        self.gamma = 0.67
        self.learning_rate = 0.1
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.model = self.built_model()
    
    def update_epsilon(self, episode):
        self.epsilon = self.min_epsilon + (self.max_epsilon-self.min_epsilon)*np.exp(-self.decay_epsilon*episode)
    
    def built_model(self):
        model = Sequential()
        model.add(Conv2D(32, (8,8), activation='relu', input_shape=self.state_size,
                        data_format='channels_last'))
        model.add(MaxPool2D(pool_size=(4,4)))
        model.add(Conv2D(64, (4,4), activation='relu'))
        model.add(MaxPool2D(pool_size=(2,2)))
        model.add(Conv2D(128, (2,2), activation='relu'))
        model.add(MaxPool2D(pool_size=(2,2)))
        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self.action_size, activation='softmax', name='predict'))

        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate), metrics=['accuracy'])

        return model

    def act(self, state):
        if random.random() > self.epsilon:
            return self.model.predict(state)[0]

        return env.action_space.sample()
    
    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self):
        minibtach = random.sample(self.memory, BATCH_SIZE)

        for state, action, reward, next_state, done in minibtach:
            target = reward 
            if not done:
                 target = reward + self.gamma*np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)    
            target_f[0][np.argmax(action)] = target

            self.model.fit(state, target_f, epochs=1, verbose=0,
                            batch_size=200)

    def save_weights(self, path):
        self.model.save_weights(path)
    
    def load_weights(self, path):
        self.model.load_weights(path)


# %%
def preprocess_image(frame):
    gray_frame = rgb2gray(frame)

    gray_frame = gray_frame[8:-12, 4:-12]/255

    return cv2.resize(gray_frame, STATE_SIZE[:-1])

def stack_frames(stacked_frames, frame, is_new_episode):
    preprossed_frame = preprocess_image(frame)

    if is_new_episode:
        stacked_frames = deque([preprossed_frame for _ in range(STACK_FRAME_SIZE)], maxlen=STACK_FRAME_SIZE)
    
    else:
        stacked_frames.append(preprossed_frame)

    stacked_state = np.stack(stacked_frames, axis=2)
    return stacked_state, stacked_frames
    

# %%
env = retro.make(game='SpaceInvaders-Atari2600')
agent = DQNetork(STATE_SIZE, env.action_space.n)

# %%
for episode in range(MAX_EPISODES):
    state = env.reset()
    # print(state.shape)
    # print()
    episode_reward = 0
    is_new_episode = True
    stacked_frames = []

    for step in range(MAX_STEPS):
        # env.render()
        state, stacked_frames = stack_frames(stacked_frames, state, is_new_episode)

        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)

        episode_reward += reward

        is_new_episode = False
        next_stacked_state, _ = stack_frames(stacked_frames, next_state, is_new_episode)
        agent.memorize(state, action, reward, next_stacked_state, done)

        state = next_state

        if done:
            break    
    
    print('Episode: {}'.format(episode),
          'Total reward: {}'.format(episode_reward),
          'Explore e: {}'.format(agent.epsilon))

    if (len(agent.memory) > BATCH_SIZE):
        agent.replay()
    
    if episode % 50 == 0:
        agent.save_weights('weights/weights-dqn-episode(%d)-reward(%d).h5' %(episode, episode_reward))

    agent.update_epsilon(episode)

