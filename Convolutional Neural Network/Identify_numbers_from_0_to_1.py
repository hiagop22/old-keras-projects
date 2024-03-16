#!usr/bin/env python3
#-*- coding: utf-8 -*-
"""
    Created on 04-24-2020 / 22:46:02
    @author: Hiago dos Santos
    @e-mail: hiagop22@gmail.com
"""

# %%
from keras.datasets import mnist    # datasets of 50000 training images, labed over 10 categories
from keras.models import Sequential # Utilitário que permite montar um stack de camadas
from keras.layers.convolutional import Conv2D
# Cria um filtro convolucional que envolve a matriz de dados e produz um tensor de saída
# Número de feature maps, tamanho do Kernel, shape do kernel, e função de ativação
from keras.layers.convolutional import MaxPooling2D # Cria uma amostragem da feature map, precisamos especificar o tamanho da janela de amostragem
from keras.layers import Dropout    # Função de regularização, evitando overfitting.... Só deve ser aplicado na fase de treino
from keras.layers import Flatten    # transforma a matriz em uma coluna para entrar na rede densa
from keras.layers import Dense      #Cria uma rede densamente conectada no modelo de uma PML

import numpy as np
import matplotlib.pyplot as plt

from keras.utils import np_utils
import cv2

#plots do matplolib vão ser amazenados no pc
get_ipython().magic('matplotlib inline')

from keras import backend as k

# Ajusta o formato 
# pois pode ser (1, 28,28) ou (28,28, 1)-->sendo 1 a informação do canal da imagem... A ordem importa

# Setando a ordem que o Keras irá ler nos dados armazenados na imagem
k.set_image_data_format('channels_last')
k.image_data_format()

# %%
import tensorflow as tf
import keras.backend.tensorflow_backend as tfback

print("tf.__version__ is", tf.__version__)
print("tf.keras.__version__ is:", tf.keras.__version__)

def _get_available_gpus():
    """Get a list of available gpu devices (formatted as strings).

    # Returns
        A list of available GPU devices.
    """
    #global _LOCAL_DEVICES
    if tfback._LOCAL_DEVICES is None:
        devices = tf.config.list_logical_devices()
        tfback._LOCAL_DEVICES = [x.name for x in devices]
    return [x for x in tfback._LOCAL_DEVICES if 'device:gpu' in x.lower()]

tfback._get_available_gpus = _get_available_gpus

# %%
(X_train, Y_train), (X_test, Y_test) = mnist.load_data('mnist.npz')

X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')

X_train /= 255
X_test /= 255

Y_train = np_utils.to_categorical(Y_train)  # converteu de vetor para matriz que vai rotular os dados
Y_test = np_utils.to_categorical(Y_test)    # converteu de vetor para matriz que vai rotular os dados

num_classes = Y_test.shape[1]

# %%
model = Sequential()
model.add(Conv2D(30, (5,5), activation='relu', input_shape=X_test.shape[1:],
        data_format='channels_first'))  # add a new layer, Conv2D(30 features maps, 
                    # tamanho da janela (filtro do kernel), shape dos dados de entrada, activaction_function=ReLu)
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(15, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))    # número de neurônios
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(num_classes, activation='softmax', name='predict')) # saída que também é uma rede densamente conectada
                                                                    # , saída sendo como activation function uma sofmax 
                                                                    # que é uma distribuição de probabilidade

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# optimization existentes = adam, gradiente descendente, gradiente descendente estocástico

model.summary() # show a resume of the architeture


# %%
# Callbacks
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', 
                                factor=0.1,
                                patiente=2,
                                verbose=0,
                                min_delta=0.0001,
                                cooldown=3,
                                min_lr=0)

early_stopping = EarlyStopping(monitor='val_accuracy', 
                                patience=4,
                                verbose=1,
                                restore_best_weights=True)

filepath = 'Weights/Weights.epoch:{epoch:02d}-accuracy{val_accuracy:.4f}.hdf5'
model_checkpoint = ModelCheckpoint(filepath, 
                                monitor='val_accuracy',
                                save_best_only=True)

learning_schedule = LearningRateScheduler(schedule=scheduler)                                

model.fit(X_train, Y_train, epochs=total_epochs, batch_size=200,
        callbacks=[model_checkpoint, early_stopping, reduce_lr])

# batch_size = lotes, para evitar que todos os dados sejam submetidos de uma vez a rede
# assim evita que o pc trave também por acabar a memória

# %%
scores = model.evaluate(X_test, Y_test)
scores[1]

# %%
img_pred = cv2.imread('number-four.png', 0)
plt.imshow(img_pred, cmap='gray')

# %%
# adequando as images ao formato de 28,28
img_pred = cv2.resize(img_pred, (28,28))
img_pred = img_pred.reshape(1, 1, 28,28).astype('float32')
print(img_pred.shape)
img_pred /= 255
print(img_pred)

# %%
pred = model.predict_classes(img_pred)
pred_proba = model.predict_proba(img_pred)
print(pred[0], 'com confiança de', pred_proba[0][pred[0]]*100)