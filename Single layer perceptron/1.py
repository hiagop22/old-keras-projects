#!usr/bin/env python3
#-*- coding: utf-8 -*-
"""
    Created on 03-28-2020 / 20:01:36
    @author: Hiago dos Santos
    @e-mail: hiagop22@gmail.com
"""

import numpy as np

# Define the number of epochs and the number of sample (q)
numEpochs = 10
q = 6

# Atributos (ph and weight)
ph = np.array([6.8, 4.7, 5.2, 3.6, 2.9, 4.2])
weight = np.array([113, 122, 107, 98, 115, 120])

# Bias
bias = 1

# Entrada e saida do perceptron (apple: -1, orange: 1)
X = np.stack((ph, weight))
Y = np.array([-1, 1, -1, -1, 1, 1])

# Parameters
eta = 0.01

# Weight vector
W = np.zeros([1, 3])

# Store errors
e = np.zeros(6)


def ativactionFuction(valor):
    # bipolar step
    if valor < 0:
        return -1
    return 1

for j in range(numEpochs):
    for k in range(q):
        # Insert Bias into the input vector
        Xb = np.hstack((bias, X[:,k]))
        
        # Calculate the induced field
        V = np.dot(W, Xb)

        # Calculate the perceptron output
        Yr = ativactionFuction(V)

        # Calculate the error
        e[k] = Y[k] - Yr

        # Training of perceptron
        W = W + eta*e[k]*Xb

print('Errors Vector (e): '+ str(e))
