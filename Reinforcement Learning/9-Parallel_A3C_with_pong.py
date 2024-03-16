# %% [code]
#!usr/bin/env python3
#-*- coding: utf-8 -*-
"""
    Created on 05-07-2020 / 20:07:07
    @author: Hiago dos Santos
    @e-mail: hiagop22@gmail.com
"""

# %% [code]
import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import cv2
import gym
import random
import numpy as np
# from collections import deque
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
from keras.optimizers import Adam, RMSprop
from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow.keras.backend import set_session
import threading
from threading import Thread, Lock
import time

# %% [code]
# Configure Keras, TensorFlow session and Graph
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
set_session(sess)
K.set_session(sess)
graph = tf.get_default_graph()

# %% [code]
# Constants
STACK_SIZE = 4
FRAME_SIZE = (80,80)
MAX_EPISODES = 2000
BATCH_SIZE = 1000   # each one is a time step, not a episode. Change it if you have a GPU
STATE_SIZE = (STACK_SIZE, *FRAME_SIZE)

# %% [code]
class PGAgent(object):
    def __init__(self, env_name):
        self.env_name = env_name
        self.env = gym.make(env_name)
        self.state_size = STATE_SIZE
        self.action_size = self.env.action_space.n
        self.learning_rate = 0.000025
        self.gamma = 0.99
        self.max_average = -21.0 # First max_average Specific for Pong
        # self.states, self.actions, self.rewards, = [], [], [] # changing it to local environment, instead global
        self.scores, self.episodes, self.average = [], [], []
        # self.image_memory = np.zeros(self.state_size) # the same reason as above
        self.actor, self.critic = self.build_model()
        # self.model.summary()
        self.save_path = 'models'
        self.path = 'PG_env:{}_lr:{}'.format(self.env, self.learning_rate)
        self.actor_name = os.path.join(self.save_path, self.path + '_actor')
        self.critic_name = os.path.join(self.save_path, self.path + '_critic')
        self.check_or_create_save_path()

        self.episode = 0 # used to track the episodes total count of episodes played through all environments
        self.lock = Lock() # lock all to update parameters without other thread interruption
        
        # threaded predict function
        self.actor._make_predict_function()
        self.critic._make_predict_function()

        global graph
        graph = tf.get_default_graph()

    def build_model(self):
        actor = Sequential()
        actor.add(Flatten(input_shape=self.state_size))
        actor.add(Dense(512, activation='elu', init='he_uniform'))
        actor.add(Dense(self.action_size, activation='softmax', init='he_uniform'))
        opt = RMSprop(lr=self.learning_rate)
        actor.compile(loss='categorical_crossentropy', optimizer=opt)

        critic = Sequential()
        critic.add(Flatten(input_shape=self.state_size))
        critic.add(Dense(512, activation='elu', init='he_uniform'))
        critic.add(Dense(1, init='he_uniform'))
        opt = RMSprop(lr=self.learning_rate)
        critic.compile(loss='mse', optimizer=opt)

        return actor, critic

    def calculate_average(self, score):
        self.scores.append(score)
        self.average.append(sum(self.scores[-50:]) / len(self.scores[-50:]))

        return self.average[-1]
    
    def save(self):
        self.actor.save(self.actor_name + '.h5')
        self.critic.save(self.critic_name + '.h5')
    
    def load(self):
        self.actor.load_weights(self.actor_name + '.h5')
        self.critic.load_weights(self.critic_name + '.h5')

    def check_or_create_save_path(self):
        if not os.path.exists(self.save_path): os.makedirs(self.save_path)
    
    # def process_frame(self, frame, image_memory):
    #     if image_memory.shape == (1, *FRAME_SIZE):
    #         image_memory = np.squeeze(image_memory)

    #     frame = frame[35:195]
    #     frame = frame[::2, ::2, :]  # decrease the resolution, but without loss information

    #     new_frame = cv2.resize(frame, FRAME_SIZE)
    #     new_frame = cv2.cvtColor(new_frame, cv2.COLOR_RGB2GRAY)

    #     new_frame = np.array(new_frame).astype(np.float32) / 255.0

    #     return new_frame

    def reset(self, env):
        image_memory = np.zeros(self.state_size)
        frame = env.reset()
        for x in range(STACK_SIZE):
            state = self.stack_frames(frame, image_memory)

        return state
    
    def step(self, action, env, image_memory):
        next_state, reward, done, _ = env.step(action)
        next_state = self.stack_frames(next_state, image_memory)

        return next_state, reward, done
    
    def stack_frames(self, frame, image_memory):
        if image_memory.shape == (1, *self.state_size):
            image_memory = np.squeeze(image_memory)

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
        image_memory = np.roll(image_memory, 1, axis = 0)

        # inserting new frame to free space
        image_memory[0,:,:] = new_frame

        # show image frame   
        #self.imshow(self.image_memory,0)
        #self.imshow(self.image_memory,1)
        #self.imshow(self.image_memory,2)
        #self.imshow(self.image_memory,3)
        return np.expand_dims(image_memory, axis=0)

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
        prob = self.actor.predict(state)[0]
        action = np.random.choice(self.action_size, p=prob)

        return action
    
    # def memorize(self, stacked_state, action, reward):
    #     self.states.append(stacked_state)
    #     act = np.zeros(self.action_size)
    #     act[action] = 1
    #     self.actions.append(act)
    #     self.rewards.append(reward)

    def replay(self, states, actions, rewards):
        #reshape memory to appropriate shape for trainning
        states = np.vstack(states)
        actions = np.vstack(actions)
        # Compute discounted rewards
        discounted_r = self.discount_and_normalize_reward(rewards)
        
        # get critic network predictions
        values = self.critic.predict(states)[:, 0]
        # comput advantages
        advantages = discounted_r - values

        # training actor and critic networks
        self.actor.fit(states, actions, sample_weight=advantages, epochs=1, verbose=0)
        self.critic.fit(states, discounted_r, epochs=1, verbose=0)
        # reset trainning memory
        # self.states, self.actions, self.rewards = [], [], []

    def run(self):
        for episode in range(MAX_EPISODES):
            state = self.reset(self.env)
            score, saving = 0, ''

            # Instantiate games memory
            states, actions, rewards = [], [], []
            while True:
                # self.env.render()
                # Actor picks an action
                action = self.act(state)
                next_state, reward, done = self.step(action, self.env, state)
                
                # self.memorize(stacked_state, action, reward)
                states.append(state)
                action_p = np.zeros([self.action_size])
                action_p[action] = 1
                actions.append(action_p)
                rewards.append(reward)

                #update current state
                state = next_state
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
    
                    self.replay(states, actions, rewards)    
                    break

        self.env.close()

    
    def train(self, n_threads):
        self.env.close()
        # instatiate one environment per thread
        envs = [gym.make(self.env_name) for _ in range(n_threads)]

        # Create threads
        threads = [threading.Thread(
                target=self.train_treading,
                daemon=True,
                args=(self, envs[i], 
                i)) for i in range(n_threads)]

        for t in threads:
            time.sleep(2)
            t.start()
        
        for t in threads:
            time.sleep(10)
            t.join()

    def train_treading(self, agent, env, thread):
        global graph
        with graph.as_default():
            while self.episode < MAX_EPISODES:
                # Reset episodes
                score, saving = 0, ''
                state = self.reset(env)
                
                # Instantiate or reset games memory
                states, actions, rewards = [], [], []
                while True:
                    action = agent.act(state)
                    next_state, reward, done = self.step(action, env, state)

                    states.append(state)
                    action_p = np.zeros([self.action_size])
                    action_p[action] = 1
                    actions.append(action_p)
                    rewards.append(reward)

                    score += reward
                    state = next_state

                    if done:
                        break
                
                self.lock.acquire()
                self.replay(states, actions, rewards)
                self.lock.release()
                
                # Update episode count
                with self.lock:
                    average = self.calculate_average(score)
                    # saving best models
                    if average >= self.max_average:
                        self.max_average = average
                        self.save()
                        saving = 'SAVING'

                    else:
                        saving = ''
                    print('Episode: {}/{}, Thread: {}, Score: {}, Average: {:.2f} {}'
                    .format(self.episode, MAX_EPISODES, thread, score, average, saving))
                    
                    if (self.episode < MAX_EPISODES):
                        self.episode += 1
            env.close()

# %% [code]
# np.set_printoptions(threshold=sys.maxsize)
import multiprocessing
n_workers = multiprocessing.cpu_count()
env_name = 'Pong-v0' 
agent = PGAgent(env_name)
agent.train(n_threads = n_workers)
# agent.test()