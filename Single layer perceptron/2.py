#!usr/bin/env python3
#-*- coding: utf-8 -*-
"""
    Created on 03-29-2020 / 11:25:48
    @author: Hiago dos Santos
    @e-mail: hiagop22@gmail.com
"""

import numpy as np

epochs = 50
q = 100
eta = 0.6
bias = 1

# Atributtes
samples = 50

mu, sigma = 10, 1
c1 = np.random.normal(mu, sigma, samples)
mu, sigma = -10, 1
c2 = np.random.normal(mu, sigma, samples)

X = np.concatenate([c1, c2])
Xb = np.stack((np.ones(100), X))
print(Xb)
Y = np.concatenate([np.zeros(50), np.ones(50)])

W = np.zeros([1, 2])
e = np.zeros(100)

def activactionFuction(value):
    if value < 0:
        return 0
    return 1


for j in range(epochs):
    for k in range(q):
        V = np.dot(W, Xb[:,k])

        Yr = activactionFuction(V)
        e[k] = Y[k] - Yr
        W = W + eta*e[k]*Xb[:,k]

print('Erro (e): ' + str(e))
        