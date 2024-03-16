#!usr/bin/env python3
#-*- coding: utf-8 -*-
"""
    Created on 05-17-2020 / 22:24:37
    @author: Hiago dos Santos
    @e-mail: hiagop22@gmail.com
"""

# %%
import tensorflow as tf

a = tf.constant(5)
b = tf.constant(3)
c = tf.constant(2)

# d é um tensor de multiplicação do tipo inteiro
d = tf.multiply(a, b)
e = tf.add(b, c)
f = tf.subtract(d, e)

# abrindo uma sessão 
sess = tf.Session()
saida = sess.run(f) # Colocando a sessão para rodar
sess.close()    # Quando fecho a sessão os resultados ficam salvos, mas os
                # recursos ficam livres

print(saida)

# Grafos:
# conjunto de arestas e nós (valores que vão fluir no grafo):
# Existe uma relação de dependência 