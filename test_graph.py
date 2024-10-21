#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

name = ['dir1', 'dir2', 'dir3', 'dir4', 'combined']

for exp in name:
    data = np.load('/home/sx-zhang/work/CNN-LSTM-master/{}.npy'.format(exp), allow_pickle=True)
    accuracy = data[0]
    epochs = np.arange(100)

    plt.figure(figsize=(6,4))
    plt.plot(epochs, accuracy)
    plt.ylim([0,1.1])
    plt.legend()
    plt.savefig('{}.png'.format(exp))
