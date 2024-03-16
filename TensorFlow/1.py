#!usr/bin/env python3
#-*- coding: utf-8 -*-
"""
    Created on 05-17-2020 / 22:13:32
    @author: Hiago dos Santos
    @e-mail: hiagop22@gmail.com
"""

# %%
import tensorflow as tf

frase = tf.constant('Ola, mundo!')
with tf.Session() as sess :
    rodar = sess.run(frase)
print(rodar)
print(rodar.decode('UTF-8'))
# %%
# Antes de eu abrir a sessão e colocar para rodar a variável, ela permanece
# como um tensor constante do tipo string    
# O valor só vai realmente fluir para dentro do tensor quando eu colocar para 
# rodar
print(frase)
print(tf.Session().run(frase))
