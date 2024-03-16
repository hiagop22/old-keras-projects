#!usr/bin/env python3
#-*- coding: utf-8 -*-
"""
    Created on 05-17-2020 / 23:16:27
    @author: Hiago dos Santos
    @e-mail: hiagop22@gmail.com
"""

# %%
import tensorflow as tf
# O tensorflow enxerga um vetor como um tensor de dimensão 1,
# mas precisa de ter pelo menor dimensão 2 para ele poder ser multplicado

a = tf.constant([[1,2,3],[4,5,6]])
b = tf.constant([[0,0],[1,0],[0,1]])

resultado = tf.matmul(a, b)
print(tf.Session().run(resultado))

x = tf.constant([0,1,0])
print(tf.Session().run(x))

print(x.shape)

# Aumenta 1 dimensão no vetor x para que ele possa ser multplicado
x = tf.expand_dims(x,1)

resultado2 = tf.matmul(a, x)
print(tf.Session().run(resultado2))