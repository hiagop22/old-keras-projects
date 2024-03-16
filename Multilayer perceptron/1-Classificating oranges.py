#!usr/bin/env python3
#-*- coding: utf-8 -*-
"""
    Created on 03-29-2020 / 20:24:34
    @author: Hiago dos Santos
    @e-mail: hiagop22@gmail.com
"""

import numpy as np
import matplotlib.pyplot as plt

# Atributte Vectors
peso = np.array([113, 122, 107,  98, 115, 120, 104, 108, 117, 101, 112, 106, 116])
pH   = np.array([6.8, 4.7, 5.2, 3.6, 2.9, 4.2, 6.3, 4.0, 6.3, 4.2, 5.6, 3.1, 5.0])

numEpochs = 40
q = 13

eta = 0.005  # learning rate
m = 2       # number of neurons in input layer
n = 2       # number of neurons in middle layer
l = 1       # number of neurons in output layer

W1 = np.random.random((n, m + 1))
W2 = np.random.random((l, n + 1))

E = np.zeros(q)
etm = np.zeros(numEpochs)

bias = 1
X = np.vstack((peso, pH))
Yd = np.array([-1, 1, -1, 1, 1, 1, -1, 1, -1, 1, -1, 1, -1])

for i in range(numEpochs):
    for k in range(q):
        Xb = np.hstack((bias, X[:,k]))

        O1 = np.tanh(W1.dot(Xb))
        O1b = np.insert(O1, 0, bias)
        Y = np.tanh(W2.dot(O1b))
        
        e = Yd[k] - Y

        E[k] = (e.transpose().dot(e))/2

        # backpropagation
        delta2 = np.diag(e).dot((1 - Y*Y))
        Vdelta2 = (W2.transpose()).dot(delta2)
        delta1 = np.diag(1 - np.dot(np.diag(O1b),O1b)).dot(Vdelta2)

        W1 = W1 + eta*(np.outer(delta1[1:], Xb))
        W2 = W2 + eta*(np.outer(delta2, O1b))

    etm[i] = E.mean()

plt.plot(etm)
plt.show()