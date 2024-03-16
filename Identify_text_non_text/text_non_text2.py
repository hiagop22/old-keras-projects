# %%
# from keras.l typing_extensions import Concatenate
from os import name
from keras import optimizers
from keras.backend import flatten, log
from keras.layers import concatenate
from keras.layers.convolutional import Deconv2D
from keras.preprocessing import image
import tensorflow as tf
# from tensorflow.keras.layers import GlobalAver
from tensorflow.keras.utils import plot_model
from keras import Model
from keras.optimizers import SGD
from keras.layers import Conv2DTranspose, MaxPool2D, Dense, Flatten, Dropout
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

import numpy as np
import cv2 as cv
%matplotlib inline
# %%
class MSPNet(object):
    def __init__(self):
        self.image_size = (224,224, 3)
        self.lr = 1e-1

        self.model = self.build_model()

    def build_model(self):
        vgg16_model = VGG16(weights='imagenet', include_top=False, 
                        input_shape=self.image_size)

        x = Flatten(name='flatten')(vgg16_model.layers[-1].output)
        x = Dropout(rate=0.5)(x)
        x = Dense(4096, name='fc1')(x)
        x = Dense(4096, name='fc2')(x)
        output = Dense(2, activation='softmax', name='fc1')(x)

        model = Model(inputs=vgg16_model.input, outputs=output)
        plot_model(model, to_file='network.png')
        
        # opt = SGD(lr=self.lr)

        # model.compile(loss=tf.keras.BinaryCrossentropy(from_logits=True), optimizer=)


test = MSPNet()
# %%
x = np.arange(5).reshape(1, 5)

print(x)
# [[ 0  1  2  3  4]
#   [ 5  6  7  8  9]
#   [10 11 12 13 14]
#   [15 16 17 18 19]]
y = np.arange(5, 15).reshape(1, 10)
y

# z = np.arange(20, 30).reshape(3, 2, 5)
# w = np.arange(20, 30).reshape(3, 2, 5)
# print(y)
# # [[20 21 22 23 24]
# #  [25 26 27 28 29]]
x = tf.keras.layers.Concatenate()([x, y])
x
# tf.keras.layers.Flatten()(x)

# %%
def adapmaxpooling(input,outside=4):
    x_shape = np.shape(input)
    batch, dim1, dim2, channels = x_shape
    stride=np.floor(dim1/outside).astype(np.int32)
    kernels=dim1-(outside-1)*stride
    print('kernel e stride')
    print(kernels)
    print(stride)
    adpooling = MaxPool2D(pool_size=(kernels, kernels), strides=(stride, stride))(x)

    return adpooling

input_shape = (1, 4, 4, 1)
x = tf.random.normal(input_shape)
x_ada = adapmaxpooling(x)
print('teste1: ')
print(x_ada.shape)
print(x_ada)

print('teste2: ')
y = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(1,1))(x)
print(y.shape)
print(y)