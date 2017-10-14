# -*- coding: utf-8 -*-
#良好拟合模型就是在训练集和测试集上表现都良好的模型。
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from matplotlib import pyplot
from numpy import array
#return training data
def get_train():
    seq = [[0.0, 0.1], [0.1,0.2], [0.2, 0.3], [0.3, 0.4], [0.4, 0.5]]
    seq = array(seq)
    X, y = seq[:, 0], seq[:, 1]
    X = X.reshape(len(X), 1, 1)
    return X, y

# return val data
def get_val():
    seq = [[0.5, 0.6], [0.6, 0.7], [0.7, 0.8], [0.8, 0.9], [0.9, 1.0]]
    seq = array(seq)
    X, y = seq[:, 0], seq[:, 1]
    X = X.reshape(len(X), 1, 1)
    return X, y

#define model
model = Sequential()
model.add(LSTM(10, input_shape=(1, 1)))
model.add(Dense(1, activation='linear'))

#compile model
model.compile('adam', 'mse')

#fit model
X, y = get_train()
valX, valY = get_val()
history = model.fit(X, y, epochs=800, validation_data=(valX, valY), shuffle= False)

#plot train and val loss
pyplot.plot(history.history['loss'])
pyplot.plot(history.history['val_loss'])
pyplot.title('model train vs val loss')
pyplot.ylabel('loss')
pyplot.xlabel('epoch')
pyplot.legend(['train', 'val'], loc='upper right')
pyplot.show()