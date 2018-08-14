#!/usr/bin/env python3

#Thanks to sagekg
#Debug this!!

import numpy as np
from PIL import Image
from tqdm import tqdm, trange
import argparse, glob, csv
from os import getcwd

def get_input_data(p, i, j, kernel_size, h, w):
    min_i = int(i-(kernel_size-1)/2)
    max_i = int(i+(kernel_size-1)/2)
    min_j = int(i-(kernel_size-1)/2)
    max_j = int(i+(kernel_size-1)/2)

    data = []
    for k in range(min_i, max_i):
        data.append([])
        for l in range(min_j, max_j):
            # if k != i and l != j:
            if k<0 or l<0 or k>=h or l>=w:
                data[k-min_i].append(1)
            elif p[k][l] >= 254:
                data[k-min_i].append(0
                )
            else:
                data[k-min_i].append(1)

    return np.array(data)

if __name__ == "__main__":
    path = getcwd() + '/'

    # open('indata.csv', 'w')
    # indata = csv.writer(open('indata.csv', 'a'))
    #
    # open('outdata.csv', 'w')
    # outdata = csv.writer(open('outdata.csv', 'a'))

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', action='store',default=path+'input_maps/',
                        dest='inpath', help="set the input data folder")
    parser.add_argument('-o', action='store', default=path+'label_maps/',
                        dest='outpath', help="set the output data folder")

    results = parser.parse_args()
    inpath = results.inpath
    outpath = results.outpath

    #Size of one side of the scanning window
    kernel_size = 31

    #get ordered lists of files
    infiles = sorted(glob.glob(inpath + '*'))
    outfiles = sorted(glob.glob(outpath + '*'))
    print(infiles)
    assert len(infiles) == len(outfiles), "Mismatched number of files"

    for i in trange(len(infiles), position=0):
        I = Image.open(infiles[i])
        w, h = I.size
        p = np.asarray(I).astype('int')

        output_im = Image.open(outfiles[i])
        output_data = np.asarray(output_im)

        input_data_len = kernel_size*kernel_size

        input_data = []
        output_list = []

        for i in trange(h, position=1):
            for j in range(w):
                if p[i][j] >= 254 and output_data[i][j] != 0:
                    input_data.append(get_input_data(p, i, j, kernel_size, h, w))
                    output_list.append(output_data[i][j])
                    # indata.writerow(get_input_data(p, i, j, kernel_size, h, w))
                    # outdata.writerow([output_data[i][j]])

        np.save("indata",np.array(input_data))
        np.save("outdata", np.array(output_list))
