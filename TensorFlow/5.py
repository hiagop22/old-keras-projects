#!usr/bin/env python3
#-*- coding: utf-8 -*-
"""
    Created on 05-18-2020 / 22:23:52
    @author: Hiago dos Santos
    @e-mail: hiagop22@gmail.com
"""
# %%
import tensorflow as tf

a = tf.Variable(2)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init) # rodando o inicilalizador de variáves
    # para variaáveis ele espera que eu já tenha alocado espaço
    # de memória para essas variáveis
    variable = sess.run(a)

print(variable)

# %%
matriz = tf.random_normal((3,5), 0, 1) # sape, media, desvio
variavel2 = tf.Variable(matriz)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    rodar = sess.run(variavel2)

print(rodar)