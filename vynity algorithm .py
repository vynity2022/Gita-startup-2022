#helpers:
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import requests
import datetime
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import math
import os
import datetime
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from cryptocmd import CmcScraper
from fastquant import get_crypto_data
from keras.callbacks import EarlyStopping















#obtaining data
df = get_crypto_data("ETH/USDT", "2014-05-01", "2022-09-22") #prediction of ethereum





























#split for test and train
split = 0.85
i_split = int(len(df) * split)
cols = ["close", "volume"]
data_train = df.get(cols).values[:i_split]
data_test = df.get(cols).values[i_split:]
#testing:
len_train = len(data_train)
len_test = len(data_test)
# print(len(df), len_train, len_test)

sequence_length = 50; input_dim = 2; batch_size = 32; epochs = 2



model = tf.keras.Sequential([
    tf.keras.layers.LSTM(100, input_shape=(sequence_length-1, input_dim), return_sequences=True),
    tf.keras.layers.Dropout(.2),
    tf.keras.layers.LSTM(100, return_sequences=True),
    tf.keras.layers.LSTM(100, return_sequences=False),
    tf.keras.layers.Dropout(.2),
    tf.keras.layers.Dense(1, activation='linear')
])

model.compile(optimizer='adam',
              loss='mse',
              metrics=['accuracy'])

def normalise_windows(window_data, single_window=False):
    '''Normalise window with a base value of zero'''
    normalised_data= []
    window_data = [window_data] if single_window else window_data
    for window in window_data:
        normalised_window = []
        for col_i in range(window.shape[1]):
            normalised_col = [((float(p) / float(window[0, col_i])) - 1) for p in window[:, col_i]]
            normalised_window.append(normalised_col)
        normalised_window = np.array(normalised_window).T
        normalised_data.append(normalised_window)
    return np.array(normalised_data)

def _next_window(i, seq_len, normalise):
    '''Generates the next data window from the given index location i'''
    window = data_train[i:i+seq_len]
    window = normalise_windows(window, single_window = True)[0] if normalise else window
    x = window[:-1]
    y = window[-1, [0]]
    return x, y

def get_train_data(seq_len, normalise):
    '''Create x, y train data windows
    Warning: batch method, not generative, make sure you hvve enough memory to load data,
    otherwise use generate_training_window() method'''
    data_x = []
    data_y = []
    for i in range(len_train - seq_len +1):
        x, y = _next_window(i, seq_len, normalise)
        data_x.append(x)
        data_y.append(y)
    return np.array(data_x), np.array(data_y)


x, y = get_train_data(
    seq_len=sequence_length,
    normalise=True
)

#how many steps we need
steps_per_epoch = math.ceil((len_train - sequence_length) / batch_size)
#print(steps_per_epoch)

callbacks = [
    EarlyStopping(monitor='accuracy', patience=2)
]

model.fit(x, y, epochs=epochs, batch_size=batch_size, callbacks=callbacks)

def get_test_data(seq_len, normalise):
    data_windows = []
    for i in range(len_test - seq_len):
        data_windows.append(data_test[i:i+seq_len])

    data_windows = np.array(data_windows).astype(float)
    data_windows = normalise_windows(data_windows, single_window=False) if normalise else data_windows

    x = data_windows[:, :-1]
    y = data_windows[:, -1, [0]]
    return x,y

x_test, y_test = get_test_data(
    seq_len = sequence_length,
    normalise=True
)

model.evaluate(x_test, y_test, verbose=2)

def get_last_data(seq_len, normalise):
    last_data = data_test[seq_len:]
    data_windows = np.array(last_data).astype(float)
    data_windows=normalise_windows(data_windows, single_window=True) if normalise else data_windows
    return data_windows

last_data_2_predict_prices = get_last_data(-(sequence_length-1), False)
last_data_2_predict_prices_1st_price = last_data_2_predict_prices[0][0]
last_data_2_predict = get_last_data(-(sequence_length-1), True)
#print("***", -(sequence_length-1), last_data_2_predict.size, "***")


def predict_point_by_point(data):
    predicted = model.predict(data)
    predicted2 = np.reshape(predicted, (predicted.size,))
    return predicted2
def de_normalise_predicted(price_1st, _data):
    return(_data +1) * price_1st
predictions2 = predict_point_by_point(last_data_2_predict)
predicted_price = de_normalise_predicted(last_data_2_predict_prices_1st_price, predictions2[0])
predictions = predict_point_by_point(x_test)
def plot_results(predicted_data, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()

# def plot_results_multiple(predicted_data, true_data, prediction_len):
#     fig=plt.figure.Figure(facecolor='white')
#     ax = fig.add_subplot(111)
#     ax.plot(true_data, label='True Data')
#     for i, data in enumerate(predicted_data):
#         padding = [None for p in range(i * (prediction_len))]
#         plt.plot(padding + data, label='Prediction')
#         plt.legend()
#     plt.show()

plot_results(predictions, y_test)


def get_change(current, previous):
    if current == previous:
        return 0
    try:
        return (abs(current - previous) / previous)*100.0
    except ZeroDivisionError:
        return float('inf')
def mean(numbers):
    return float(sum(numbers)) / max(len(numbers), 1)
def last_change(ls1, ls2):
    ls3 = []
    for i in ls1:
        ls3.append(get_change(ls1[i], ls2[i]))
    ans = mean(ls3)
    return ans

# setA = set(predictions)
# setB = set(y_test)
#
# ans = len(setA & setB) / (float(len(setA | setB))) * 100
# print(ans)

print(type(predictions))
print(type(y_test))