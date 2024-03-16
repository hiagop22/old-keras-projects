#!usr/bin/env python3
#-*- coding: utf-8 -*-
"""
    Created on 05-17-2020 / 23:04:06
    @author: Hiago dos Santos
    @e-mail: hiagop22@gmail.com
"""
# %%
import tensorflow as tf

a = tf.constant(2, dtype=tf.float32)  # 1x1
b = tf.constant([3,1,5,8,6])    # 1xn
c = tf.constant([[2,0,4] , [3,5,7]])    # nxn

sess = tf.Session()
sess.run(c)

sess.close()

# %%
import numpy as np
a1 = np.array(2)
a2 = np.array([3,1,5,8,6])
a3 = np.array([[2,0,4] , [3,5,7]])

# %%
print(a)
print(a1)
