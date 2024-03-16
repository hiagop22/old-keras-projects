#!usr/bin/env python3
#-*- coding: utf-8 -*-
"""
    Created on 05-07-2020 / 20:07:07
    @author: Hiago dos Santos
    @e-mail: hiagop22@gmail.com
"""

# %%
import os
import cv2
import gym
import random
# import sys
# import pylab
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
from keras.optimizers import Adam, RMSprop
import keras.backend as k

# %%
# Constants
STACK_SIZE = 4
FRAME_SIZE = (80,80)
MAX_EPISODES = 2000
BATCH_SIZE = 1000   # each one is a time step, not a episode. Change it if you have a GPU
STATE_SIZE = (STACK_SIZE, *FRAME_SIZE)

# %%
class PGAgent(object):
    def __init__(self, env_name):
        self.env = gym.make(env_name)
        self.state_size = STATE_SIZE
        self.action_size = self.env.action_space.n
        self.learning_rate = 0.000025
        self.gamma = 0.99
        self.max_average = -21.0 # First max_average Specific for Pong
        self.states, self.actions = [], []
        self.rewards, self.scores = [], []
        self.average = []
        self.image_state = np.zeros(self.state_size)
        self.model = self.build_model()
        # self.model.summary()
        self.save_path = 'models'
        self.path = 'PG_env:{}_lr:{}'.format(self.env, self.learning_rate)
        self.model_name = os.path.join(self.save_path, self.path)
        
        self.check_or_create_save_path()
    
    def build_model(self):
        model = Sequential()
        # model.add(Conv2D(32, (8,8), activation='relu', input_shape=self.state_size,
                        # init='he_uniform'))
        # model.add(MaxPool2D(pool_size=(4,4)))
        # model.add(Conv2D(64, (4,4), activation='relu'))
        # model.add(MaxPool2D(pool_size=(2,2)))
        # model.add(Dropout(0.2))
        model.add(Flatten(input_shape=self.state_size))
        model.add(Dense(512, activation='elu', init='he_uniform'))
        # model.add(Dense(32, activation='relu', init='he_uniform'))
        model.add(Dense(self.action_size, activation='softmax', init='he_uniform'))
        opt = RMSprop(lr=self.learning_rate)
        model.compile(loss='categorical_crossentropy', optimizer=opt)

        return model

    def calculate_average(self, scores):
        self.scores.append(scores)
        self.average.append(sum(self.scores[-50:]) / len(self.scores[-50:]))

        return self.average[-1]
    
    def save(self):
        self.model.save(self.model_name + '.h5')
    
    def load(self):
        self.model.load_weights(self.model_name + '.h5')

    def check_or_create_save_path(self):
        if not os.path.exists(self.save_path): os.makedirs(self.save_path)
    
    def process_frame(self, frame):
        frame = frame[35:195]
        frame = frame[::2, ::2, :]  # decrease the resolution, but without loss information

        new_frame = cv2.resize(frame, FRAME_SIZE)
        new_frame = cv2.cvtColor(new_frame, cv2.COLOR_RGB2GRAY)

        new_frame = np.array(new_frame).astype(np.float32) / 255.0
        # frame[frame == 144] = 0     # set backgroung to 0
        # frame[frame == 236] = 2     # set ball to 2
        # frame[frame == 92 ] = 1     # set own cursor to 1 
        # frame[frame == 213] = 1     # set opponent cursor to 1

        return new_frame
    
    def stack_frames(self, frame):
        # processed_frame = self.process_frame(frame)
        
        # # if stacked_frames == []:
        # #     stacked_frames = deque([processed_frame for _ in range(STACK_SIZE)], maxlen=STACK_SIZE)
        # # else:
        # #     stacked_frames.append(processed_frame)
                
        # # stacked_state = np.stack(stacked_frames, axis=2)
        # # stacked_state = np.expand_dims(stacked_state, axis=0)

        # # return stacked_state, stacked_frames
        # self.image_state = np.roll(self.image_state, 1, axis=0)
        # self.image_state[0, : , :] = processed_frame
        # a = np.expand_dims(self.image_state, axis=0)

        # return a/255.0

         # croping frame to 80x80 size
        frame_cropped = frame[35:195:2, ::2,:]
        if frame_cropped.shape[0] != 80 or frame_cropped.shape[1] != 80:
            # OpenCV resize function 
            frame_cropped = cv2.resize(frame, (80, 80), interpolation=cv2.INTER_CUBIC)
        
        # converting to RGB (numpy way)
        frame_rgb = 0.299*frame_cropped[:,:,0] + 0.587*frame_cropped[:,:,1] + 0.114*frame_cropped[:,:,2]

        # convert everything to black and white (agent will train faster)
        frame_rgb[frame_rgb < 100] = 0
        frame_rgb[frame_rgb >= 100] = 255
        # converting to RGB (OpenCV way)
        #frame_rgb = cv2.cvtColor(frame_cropped, cv2.COLOR_RGB2GRAY)     

        # dividing by 255 we expresses value to 0-1 representation
        new_frame = np.array(frame_rgb).astype(np.float32) / 255.0

        # push our data by 1 frame, similar as deq() function work
        self.image_state = np.roll(self.image_state, 1, axis = 0)

        # inserting new frame to free space
        self.image_state[0,:,:] = new_frame

        # show image frame   
        #self.imshow(self.image_memory,0)
        #self.imshow(self.image_memory,1)
        #self.imshow(self.image_memory,2)
        #self.imshow(self.image_memory,3)
        return np.expand_dims(self.image_state, axis=0)


    def discount_and_normalize_reward(self, reward):
        # apply the discount and normalize it to avoid big variability os rewards
        discounted_reward = np.zeros_like(reward)
        cummulative = 0.0

        for i in reversed(range(len(reward))):
            if reward[i] != 0:
                cummulative = 0.0       # reset the summation
            cummulative = cummulative * self.gamma + reward[i]
            discounted_reward[i] = cummulative

        mean = np.mean(discounted_reward)
        std = np.std(discounted_reward)
        discounted_reward = (discounted_reward - mean)/std
        
        return discounted_reward
    
    def act(self, state):
        """ example:
        pred = np.array([0.05, 0.85, 0.1])
        action_size = 3
        np.random.choice(a, p=pred)
        result>>> 1, because it have the highest probability to be taken
        """
        prob = self.model.predict(state)[0]
        action = np.random.choice(self.action_size, p=prob)

        return action
    
    def memorize(self, stacked_state, action, reward):
        self.states.append(stacked_state)
        act = np.zeros(self.action_size)
        act[action] = 1
        self.actions.append(act)
        self.rewards.append(reward)

    def replay(self):
        #reshape memory to appropriate shape for trainning
        states = np.vstack(self.states)
        actions = np.vstack(self.actions)
        # Compute discounted rewards
        discounted_r = self.discount_and_normalize_reward(self.rewards)
        self.model.fit(states, actions, sample_weight=discounted_r, epochs=1, verbose=0)
        # reset trainning memory
        self.states, self.actions, self.rewards = [], [], []

    def train(self):
        for episode in range(MAX_EPISODES):
            frame = self.env.reset()
            score, saving = 0, ''
            for x in range(4):
                stacked_state = self.stack_frames(frame)

            while True:
                # self.env.render()
                action = self.act(stacked_state)

                next_frame, reward, done, _ = self.env.step(action)
                
                self.memorize(stacked_state, action, reward)

                frame = next_frame
                stacked_state = self.stack_frames(frame)
                score += reward

                if done:
                    average = self.calculate_average(score)

                    if average >= self.max_average:
                        saving = 'SAVING'
                        self.max_average = average
                        self.save()
                    else:
                        saving = ''

                    print('Episode: {}/{} Score: {} Average: {:.2f} {}'
                    .format(episode, MAX_EPISODES, score, average, saving))
    
                    self.replay()
    
                    break

        
        self.env.close()
                
# %%
# np.set_printoptions(threshold=sys.maxsize)
env_name = 'Pong-v0' 
agent = PGAgent(env_name)
agent.train()
# agent.test()