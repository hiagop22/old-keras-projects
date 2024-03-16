
#!usr/bin/env python3
#-*- coding: utf-8 -*-
"""
    Created on 04-01-2020 / 11:11:56
    @author: Hiago dos Santos
    @e-mail: hiagop22@gmail.com
"""

import numpy as np
import matplotlib.pyplot as plt

numEpocs = 2000
q = 8

X = np.array((range(10),
            range(9, -1, -1),
            [0,9,1,8,2,7,3,6,4,5],
            [4,5,6,3,2,7,1,8,0,9],
            [3,8,2,7,1,6,0,5,9,4],
            [1,6,0,7,4,8,3,9,2,5],
            [2,1,3,0,4,9,5,8,6,7],
            [9,4,0,5,1,6,2,7,3,8]))

Yd = np.array(([-1,1],
              [1,1],
              [-1,1],
              [1,-1],
              [1,1],
              [1,-1],
              [-1,1],
              [-1,-1]))

# Neurons properties
m = 10
n = 4
l = 2
etas = (0.05, 0.5)
W1 = (0.5 + 0.5)*np.random.rand(n, m + 1) - 0.5
W2 = (0.5 + 0.5)*np.random.rand(l, n + 1) - 0.5
bias = 1

Etm = np.zeros(numEpocs)
E = np.zeros(q)

plt.figure()
for eta in range(len(etas)):
    for epoch in range(numEpocs):
        for sample in range(q):
            Xb = np.insert(X[sample,:], 0, bias)

            O1 = np.tanh(W1.dot(Xb))
            O1b = np.insert(O1, 0, bias)
            Y = np.tanh(W2.dot(O1b))

            e = Yd[sample] - Y
            E[sample] = (e.transpose()).dot(e)/2

            # back-propagation
            delta2 = (np.diag(e)).dot(1 - Y*Y)
            Vdelta2 = (W2.transpose()).dot(delta2)
            delta1 = (np.diag(1 - np.dot(np.diag(O1b),O1b))).dot(Vdelta2)

            W1 = W1 + etas[eta]*np.outer((delta1[1:]).transpose(), Xb)
            W2 = W2 + etas[eta]*np.outer(delta2, O1b)
        Etm[epoch] = E.mean()
    plt.subplot(220+eta+1)
    plt.plot(Etm)
plt.show()
