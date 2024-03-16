#!usr/bin/env python3
#-*- coding: utf-8 -*-
"""
    Created on 05-22-2020 / 22:57:48
    @author: Hiago dos Santos
    @e-mail: hiagop22@gmail.com
"""

# %% 
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
# tf.compat.v1.reset_default_graph()

# %%
# with tf.get_default_graph():
a = tf.constant(2, dtype=tf.float32, name='tensor_a')
b = tf.constant(4, dtype=tf.float32, name='tensor_b')
d = tf.constant(4, dtype=tf.float32, name='tensor_c')
c = tf.add(a, b) + d

# %%
with tf.compat.v1.Session() as sess:
    # Save the current graph used in this session
    writer = tf.compat.v1.summary.FileWriter('graph1', sess.graph)
    saida = sess.run(c)

print(f'SAIDA>> {saida}')

# To run tensorboard: 
# Example: tensorboard --logdir=graph1 

# %%
