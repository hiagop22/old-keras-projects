#!usr/bin/env python3
#-*- coding: utf-8 -*-
"""
    Created on 04-29-2020 / 11:44:01
    @author: Hiago dos Santos
    @e-mail: hiagop22@gmail.com
"""

# %%
import math
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

import plotly.offline as py
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt

# %%
py.init_notebook_mode(connected=True)
%matplotlib inline

# %%
df = pd.read_csv(filepath_or_buffer='Data/data_bitcoin.csv', index_col='Date')

# %%
df = df[906:-1] # Since Jul 2016 - Now
# df.head(4)

# %%
df.shape
# df.tail()

# %%
btc_trace = go.Scatter(x=df.index, y=df['Close'], name='price')
py.iplot([btc_trace])

# %%
values = df.Close.values.reshape([df.shape[0], 1])

# %%
scaler = StandardScaler()
scaled_values = scaler.fit_transform(values)

# %%
scaled_values

# %%
test_size = 30
train_size = len(scaled_values)-test_size 
train, test = scaled_values[:train_size], scaled_values[train_size:]
train.shape

# %%
fig = plt.figure(figsize=(15, 10))
ax = plt.subplot(211)
ax.set_title('Dados de treinamento(' + str(train_size) + 'dias)', loc='left', fontsize=16)
plt.plot(train)

ax = plt.subplot(212)
ax.set_title('Dados de teste(' + str(test_size) + 'dias)', loc='left', fontsize=16)
plt.plot(test)

# %%
def dataset_with_look_back(dataset, look_back):
    X, Y = [], []

    for a in range(len(dataset) - look_back):
        X.append(dataset[a: look_back+a,0])
        Y.append(dataset[look_back+a,0])

    return np.array(X), np.array(Y)

look_back = 5
print(train[:10])
print(test[:10])

X_train, Y_train = dataset_with_look_back(train, look_back)
X_test, Y_test = dataset_with_look_back(test, look_back)

# %%
print(X_train[:10])
print(Y_train[:10])
print(X_train.shape, '\n', X_test.shape)
# %%
# [samples, time_steps, features] is a requirement from the NN
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

print(X_train[:10])
X_train.shape

# %%
model = Sequential()

model.add(LSTM(30, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.1))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')

model.summary()

# %%
# Parameters
epochs = 50
batch_size = 5

# Trainning
history = model.fit(X_train, Y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=(X_test, Y_test),
                    shuffle=False)


# %%
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.title('Loss of trainning', loc='left', fontsize=16)
plt.legend()

# %%
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Invert predictions
inversed_train_predict = scaler.inverse_transform(train_predict)
inversed_Y_train = scaler.inverse_transform(Y_train)
inversed_test_predict = scaler.inverse_transform(test_predict)
inversed_Y_test = scaler.inverse_transform(Y_test)

# %%
# 

# %%
predictDates = df.tail(len(X_train)).index
actual_chart = go.Scatter(x=predictDates, y=inversed_Y_train, name='actual Price')
predict_chart = go.Scatter(x=predictDates, y=inversed_train_predict[:,0], name='predict Price')
py.iplot([predict_chart, actual_chart])

# %%
predictDates_test = df.tail(len(X_test)).index
actual_chart_test = go.Scatter(x=predictDates_test, y=inversed_Y_test, name='actual Price')
predict_chart_test = go.Scatter(x=predictDates_test, y=inversed_test_predict[:,0], name='predict Price')
py.iplot([predict_chart_test, actual_chart_test])