#!/usr/bin/env python3

#Get rid of those annoying numpy/tensorflow warnings
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

import keras
from PIL import Image
from keras.models import model_from_json
import numpy as np
from os import getcwd
from tqdm import trange, tqdm
import matplotlib.pylab as plt

# in_data = np.load("indata_for_prediction.npy")
# img_x, img_y = in_data[0].shape
# in_data = in_data.reshape(in_data.shape[0], img_x, img_y, 1)
#
# input_points = np.load("input_prediction_points.npy")

def get_input_data(p, i, j, kernel_size, h, w):
    min_i = int(i-(kernel_size-1)/2)
    max_i = int(i+(kernel_size-1)/2)
    min_j = int(j-(kernel_size-1)/2)
    max_j = int(j+(kernel_size-1)/2)

    data = []
    for k in range(min_i, max_i):
        row = [1 if k<0 or l<0 or k>=h or l>=w or p[k][l] <254 else 0 for l in range(min_j, max_j)]
        data.append(row)
        # for l in range(min_j, max_j):
        #     # if k != i and l != j:
        #     if k<0 or l<0 or k>=h or l>=w:
        #         data[k-min_i].append(1)
        #     elif p[k][l] >= 254:
        #         data[k-min_i].append(0)
        #     else:
        #         data[k-min_i].append(1)

    return np.array(data)

# load json and create model
json_file = open('convnet1.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("convnet1.h5")

loaded_model.summary()

path = 'input_maps/maze.pgm'
# data_path = path + 'etu_1_condensed/'

I = Image.open(path)
p = np.asarray(I).astype('int')
w, h = I.size
im = np.zeros((w, h), dtype=int)
im_log = np.zeros((w, h), dtype=int)

kernel_size = 61

for i in trange(h, position=1):
    for j in range(w):
        if p[i][j] >= 254:
            in_data = get_input_data(p, i, j, kernel_size, h, w)
            img_x, img_y = in_data.shape
            in_data = in_data.reshape(1, img_x, img_y, 1)
            val = loaded_model.predict(in_data, verbose=0)

            # if len(val) != 1:
            # im[i][j] = val
            if val[0][0] >= 0:
            # tqdm.write("log val: " + str(val[0][0]))
                im_log[i][j] = np.log(val[0][0])-1

plt.imshow(im_log, cmap=plt.cm.Greys, interpolation='nearest')
plt.show()

# np.save("predicted_sim_data", im)
