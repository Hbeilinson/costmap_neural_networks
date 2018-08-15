#!/usr/bin/env python3

#Get rid of those annoying numpy/tensorflow warnings
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

import keras
import numpy as np
from keras.layers import *
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from sklearn.model_selection import train_test_split
import matplotlib.pylab as plt
from liveHistCallback import *

batch_size = 20
epochs = 1000


in_data = np.load("indata.npy")
img_x, img_y = in_data[0].shape
in_data = in_data.reshape(in_data.shape[0], img_x, img_y, 1)
out_data = np.load("outdata.npy")

seed=7
np.random.seed(seed)
x_train, x_test, y_train, y_test = train_test_split(in_data, out_data, random_state=seed)

# model = unet(img_x, img_y, 1)
def cnn(img_x, img_y):
    model = Sequential()
    model.add(MaxPooling2D(pool_size=(4, 4)))
    model.add(Conv2D(64, kernel_size=(4, 4),
                        activation='relu'
                     # input_shape = (img_x, img_y, 1)
                     ))
    model.add(Dropout(0.2))
    # model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(64, kernel_size=(2, 2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    # model.add(Dense(1000, activation='relu'))
    model.add(Dense(500, kernel_initializer='normal', activation='relu'))
    # model.add(Dense(50, kernel_initializer='normal', activation='relu'))
    # model.add(Dense(5, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    #
    # model.summary()
    return model

model = cnn(img_x, img_y)
model.compile(loss=keras.losses.mean_squared_error,
                # optimizer=keras.optimizers.Adam(lr = 1e-5),
                optimizer=keras.optimizers.SGD(nesterov=True),
                metrics=['mse', 'mae'])

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
