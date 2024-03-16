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
from keras.layers import Conv2DTranspose, MaxPool2D, Dense, Flatten
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
                        pooling=None, input_shape=self.image_size)


        # removed_maxpool_model = Model(vgg16_model.input, vgg16_model.layers[-2].output)
        deconv3 = Deconv2D(128, kernel_size=(1,1), strides=1)(vgg16_model.layers[-8].output)
        deconv4 = Deconv2D(256, kernel_size=(4,4), strides=2)(vgg16_model.layers[-5].output)
        deconv5 = Deconv2D(256, kernel_size=(8,8), strides=4)(vgg16_model.layers[-2].output)
        
        def adapmaxpooling(input,outside=4):
            x_shape = np.shape(input)
            batch, dim1, dim2, channels = x_shape
            stride=np.maximum(1, np.floor(dim1/outside).astype(np.int32))
            kernels=dim1-(outside-1)*stride
            adpooling = MaxPool2D(pool_size=(kernels, kernels), strides=(stride, stride))(input)
            if outside==6*5:
                print('adpooling')
                print(adpooling.shape)

            return adpooling

        # adaptative spatial pooling 6x6 applied to a stack of feature block 1x1
        adap1_3=Flatten()(adapmaxpooling(deconv3, outside=6*1))
        adap1_3=tf.reshape(adap1_3, (1, adap1_3.shape[-1]))
        adap1_4=Flatten()(adapmaxpooling(deconv4, outside=6*1))
        adap1_4=tf.reshape(adap1_4, (1, adap1_4.shape[-1]))
        adap1_5=Flatten()(adapmaxpooling(deconv5, outside=6*1))
        adap1_5=tf.reshape(adap1_5, (1, adap1_5.shape[-1]))
        
        # adaptative spatial pooling 6x6 applied to a stack of feature block 3x3
        adap3_3=Flatten()(adapmaxpooling(deconv3, outside=6*3))
        adap3_3=tf.reshape(adap3_3, (1, adap3_3.shape[-1]))
        adap3_4=Flatten()(adapmaxpooling(deconv4, outside=6*3))
        adap3_4=tf.reshape(adap3_4, (1, adap3_4.shape[-1]))
        adap3_5=Flatten()(adapmaxpooling(deconv5, outside=6*3))
        adap3_5=tf.reshape(adap3_5, (1, adap3_5.shape[-1]))

        # adaptative spatial pooling 6x6 applied to a stack of feature block 5x5
        # adap5_3=Flatten()(adapmaxpooling(deconv3, outside=6*5))
        adap5_3=adapmaxpooling(deconv3, outside=6*5)
        adap5_3=tf.reshape(adap5_3, (1, adap5_3.shape[-1]))
        print('adap5_3')
        print(adap5_3.shape)
        # print(adapmaxpooling(deconv3, outside=6*5))
                
        # print('tensor3')
        # print(adap5_3.shape)
        
        adap5_4=Flatten()(adapmaxpooling(deconv4, outside=6*5))
        adap5_4=tf.reshape(adap5_4, (1, adap5_4.shape[1]))
        adap5_5=Flatten()(adapmaxpooling(deconv5, outside=6*5))
        adap5_5=tf.reshape(adap5_5, (1, adap5_5.shape[1]))

        # adaptative spatial pooling 6x6 applied to a stack of feature block 7x7
        adap7_3=Flatten()(adapmaxpooling(deconv3, outside=6*7))
        adap7_3=tf.reshape(adap7_3, (1, adap7_3.shape[1]))
        adap7_4=Flatten()(adapmaxpooling(deconv4, outside=6*7))
        adap7_4=tf.reshape(adap7_4, (1, adap7_4.shape[1]))
        adap7_5=Flatten()(adapmaxpooling(deconv5, outside=6*7))
        adap7_5=tf.reshape(adap7_5, (1, adap7_5.shape[1]))
        
        concatenated = concatenate([adap1_3, adap1_4, adap1_5, adap3_3, adap3_4, adap3_5,
                                    adap5_3, adap5_4, adap5_5, adap7_3, adap7_4, adap7_5])
        
        fc1 = Dense(4096, name='fc1')(concatenated)
        fc2 = Dense(4096, name='fc2')(fc1)
        output = Dense(2, activation='softmax')(fc2)

        model = Model(inputs=vgg16_model.input, outputs=output)
        plot_model(model, to_file='network.png')
        # def cross_entropy(predictions, targets, class_balancing_wight=0.67):
        #     ce = -np.sum([class_balancing_wight*targets[i]*np.log(predictions[i]+1e-9)+(1-class_balancing_wight)*(1-targets[i])*np.log(1-predictions[i]+1e-9) for i in range(len(predictions))])
        #     return ce

        # removed_maxpool_model.summary()
        # x = Deconv2D(128, (1,1), strides=1)(removed_maxpool_model)
        # x = Deconv2D(256, (4,4), strides=2)(x)
        # x = Deconv2D()
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