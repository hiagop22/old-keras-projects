#!usr/bin/env python3
#-*- coding: utf-8 -*-
"""
  Created on 04-01-2020 / 22:18:14
  @author: Hiago dos Santos
  @e-mail: hiagop22@gmail.com
"""

import numpy as np
import matplotlib.pyplot as plt

# X = (Hours slept, Hours studying), y = score obtained on test
X = np.array(([0,0],[0,1],[1,0],[1,1]), dtype=float)
Yd = np.array(([0],[1],[1],[0]), dtype=float)

# Scale units
X = X/1 # Maxima X axis
Yd = Yd/1 # Maxima Y axis

class NeuralNetwork(object):
  def __init__(self):
    self.eta = 0.025
    self.inputNumber = 2
    self.hiddenNumber = 3
    self.outputNumber = 1
    # weights
    self.W1 = np.random.randn(self.inputNumber, self.hiddenNumber)
    self.W2 = np.random.randn(self.hiddenNumber, self.outputNumber)

  def foward(self, X):
    self.Vinduced = np.tanh(np.dot(X, self.W1))
    self.Y = np.tanh(np.dot(self.Vinduced, self.W2))
    return self.Y
  
  def backward(self, X, Yd, Y):
    self.Error_out = (Yd - Y) # Ok
    self.Delta2 = (np.diagflat(self.Error_out)).dot((1 - np.diagflat(Y).dot(Y))) # Ok

    self.Vdelta = np.dot(np.tile(self.Delta2, self.hiddenNumber), self.W2) # Ok
    self.Delta1 = (np.diagflat(self.Vdelta)).dot((1 - self.Vinduced**2)) # Ok

    self.W1 += self.eta*X.T.dot(self.Delta1) # Ok
    self.W2 += self.eta*self.Vinduced.T.dot(self.Delta2) # Ok
  
  def train(self, X, Yd):
    self.Y = self.foward(X)
    self.backward(X, Yd, self.Y)
    return self.Y
  
  def predict(self, X):
    return self.foward(X)
  
  def saveWeights(self):
    np.savetxt("w1.txt", self.W1, fmt="%s")
    np.savetxt("w2.txt", self.W2, fmt="%s")

Etm = np.zeros(5000)
NN = NeuralNetwork()

for x in range(5000):
  # print("Predicted Output: \n" + str(Yd))
  # print("Actual Output: \n" + str(Y))
  Y = NN.train(X, Yd)
  Etm[x] = (Yd-Y).mean()

NN.saveWeights()
Xpredicted = np.array([[0,0],[0,1],[1,0], [1,1]], dtype=float)
print(NN.predict(Xpredicted))

plt.plot(Etm)
plt.show()