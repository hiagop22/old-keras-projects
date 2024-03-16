#!usr/bin/env python3
#-*- coding: utf-8 -*-
"""
  Created on 04-01-2020 / 22:18:14
  @author: Hiago dos Santos
  @e-mail: hiagop22@gmail.com
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.neural_network import MLPClassifier

data = load_iris()

features = data['data']
features_names = data['feature_names']
target = data['target']

tunning = {}
tunning['features'] = np.concatenate((features[:40], features[50:90], features[100:140]))
tunning['target'] = np.concatenate((target[:40], target[50:90], target[100:140]))

# print(np.concatenate((target[:40], target[50:90], target[100:140])))
validation = {}
validation['features'] = np.concatenate((features[40:50], features[90:100], features[140:150]))
validation['target'] = np.concatenate((target[40:50], target[90:100], target[140:150]))

NN = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(7,), random_state=1)
NN.fit(tunning['features'], tunning['target'])

predicted = NN.predict(validation['features'])
print('Predicted outputs: '+str(predicted))
print('Error output: '+str(validation['target'] - predicted))

