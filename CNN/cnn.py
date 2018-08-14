#!/usr/bin/env python3

#Thanks to Sage KG, https://github.com/sagekg/ebola-neural-net/blob/master/conv_net1.py

#Get rid of those annoying numpy/tensorflow warnings
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

import keras
import numpy as np
from keras.datasets import mnist
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.callbacks import History
from sklearn.model_selection import train_test_split
import matplotlib.pylab as plt

batch_size = 128
epochs = 300

img_x, img_y = 30, 30

in_data = np.load("indata.npy")
in_data = in_data.reshape(in_data.shape[0], img_x, img_y, 1)
out_data = np.load("outdata.npy")

seed=7
np.random.seed(seed)
x_train, x_test, y_train, y_test = train_test_split(in_data, out_data, test_size=.15, random_state=seed)

model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                 activation='relu',
                 input_shape=(img_x, img_y, 1)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dense(500, kernel_initializer='normal', activation='relu'))
model.add(Dense(50, kernel_initializer='normal', activation='relu'))
model.add(Dense(5, kernel_initializer='normal', activation='relu'))
model.add(Dense(1, kernel_initializer='normal'))

model.compile(loss=keras.losses.mean_squared_logarithmic_error,
              optimizer=keras.optimizers.Adam(),
              metrics=['mse', 'mae'])

class liveHist(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.mse_acc = []
        self.mae_acc = []
        self.losses = []
        self.val_losses = []

        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(3, 1, sharex=True)
        self.fig.text(.5, .04, 'Epochs', ha='center')


    def on_epoch_end(self, epoch, logs={}):
        self.mse_acc.append(logs.get('val_mean_squared_error'))
        self.mae_acc.append(logs.get('val_mean_absolute_error'))
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.x.append(self.i)
        self.i += 1

        self.ax1.cla()
        self.ax2.cla()
        self.ax3.cla()
        self.ax1.set_title('Losses')
        self.ax1.set(ylabel='Logarithmic MSE')

        self.ax2.set_title('Validation Accuracy (MSE)')
        self.ax2.set(ylabel='Mean Squared Error')

        self.ax3.set_title('Validation Accuracy (MAE)')
        self.ax3.set(ylabel='Mean Absolute Error')

        self.ax1.plot(self.x, self.losses, label="loss", color='C0')
        self.ax1.plot(self.x, self.val_losses, label="val_loss", color='C1')
        self.ax1.legend(loc="upper right")
        self.ax2.plot(self.x, self.mse_acc)
        self.ax3.plot(self.x, self.mae_acc)

        # plt.pause(.01)
        plt.draw()


history = liveHist()


model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test),
          callbacks=[history])

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
plt.savefig("lossaccplot.png", dpi='figure')

model_json = model.to_json()
with open("convnet1.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("convnet1.h5")
