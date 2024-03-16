#!usr/bin/env python3
#-*- coding: utf-8 -*-
"""
    Created on 05-18-2020 / 22:36:56
    @author: Hiago dos Santos
    @e-mail: hiagop22@gmail.com
"""

# %%
import tensorflow as tf
import numpy as np

dados_x = np.random.randn(4,8) # shape = (4,8)
dados_w = np.random.randn(8,2)

b = tf.random_normal((4,2), 0,1)

# Criando tensores placeholders, passando o tipo e o shape como parâmetros
# variáveis vazias para serem alimentadas com dados posteiormente
x = tf.placeholder(tf.float32, shape=(4,8))
w = tf.placeholder(tf.float32, shape=(8,2))
operacao = tf.matmul(x, w) + b # tf.matmul vai resultar em um tensor, logo preciso somar com um tensor

maximo = tf.reduce_max(operacao) # encontra o maior valor da matriz operacao
                                 # reduce, pois reduz a dimensão da matriz para escalar

with tf.Session() as sess:
    saida1 = sess.run(operacao, feed_dict={x:dados_x, w:dados_w})
    saida2 = sess.run(maximo, feed_dict={x:dados_x, w:dados_w})

# Placeholder: espaço livre para por exemplo entrar na rede neural para ela usar os dados
# que foram inseridos alí para treinar. Ela aprende mais rápido através de pacotes, ao invés
# de colocar todos os dados de 1 vez

# Como no tensor é backfoward, preciso informar na sessão com o que vou alimentar meus placeholders
# no formato de dicionário

# %%
print(saida1)
print(saida2)


# Alimentando com pacotes, um grande conjunto de dados
# %%

lista_x = [np.random.randn(4,8) for _ in range(5)] 
lista_w = [np.random.randn(8,2) for _ in range(5)] 

b = tf.random_normal((4,2),0,1)

x = tf.placeholder(tf.float32, shape=(4,8))
w = tf.placeholder(tf.float32, shape=(8,2))

# Perde performance por não saber qual o espaço de memória 
# que precisa alocar
# x = tf.placeholder(tf.float32, shape=(None,None))
# w = tf.placeholder(tf.float32, shape=(None,None))

operacao = tf.matmul(x,w) + b

maximo = tf.reduce_max(operacao)

lista_saida = []

with tf.Session() as sess:
    for i in range(len(lista_x)):
        saida = sess.run(maximo, feed_dict={x:lista_x[i], w:lista_w[i]})
        lista_saida.append(saida)
print(lista_saida)
