# -*- coding: utf-8 -*-
#另外，如果模型在训练集上的性能比验证集上的性能好，并且模型性能曲线已经平稳了，那么这个模型也可能欠拟合。下面就是一个缺乏足够的记忆单元的欠拟合模型的例子。
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
model.add(LSTM(1, input_shape=(1, 1)))
model.add(Dense(1, activation='linear'))

#compile model
model.compile('sgd', 'mse')

#fit model
X, y = get_train()
valX, valY = get_val()
history = model.fit(X, y, epochs=300, validation_data=(valX, valY), shuffle= False)

#plot train and val loss
pyplot.plot(history.history['loss'])
pyplot.plot(history.history['val_loss'])
pyplot.title('model train vs val loss')
pyplot.ylabel('loss')
pyplot.xlabel('epoch')
pyplot.legend(['train', 'val'], loc='upper right')
pyplot.show()