#!usr/bin/env python3
#-*- coding: utf-8 -*-
"""
    Created on 05-06-2020 / 12:58:20
    @author: Hiago dos Santos
    @e-mail: hiagop22@gmail.com
"""

# %%
import os
import gym
import random
import pylab
import numpy as np
from collections import deque
from keras.models import Model, load_model
from keras.layers import Input, Dense, Lambda, Add 
from keras.optimizers import Adam, RMSprop
from keras import backend as k
from PER import *


# %%






# %%
env_game = 'CartPole-v1'
agent = DQNetwork(env_game)
agent.run()
# agent.test()