#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 16:30:05 2017

@author: manish
"""

import numpy as np
from matplotlib import pyplot as plt 
import cv2

IMG_SIZE = 32

train_data = np.load('train_data.npy')
label_grid = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def label_decoder(label):
    index = np.argmax(label, axis=0)
    index = int(index)
    label = label_grid[index]
    return label

def plot_train_set(train_data):
    fig = plt.figure()
    for num, data in enumerate(train_data[:12]):
        img = data[0]
        label = label_decoder(data[1])
        y = fig.add_subplot(3,4,num+1)
        y.imshow(img)
        plt.title(label)
        y.axes.get_xaxis().set_visible(False)
        y.axes.get_yaxis().set_visible(False)
    plt.show()
  
def reshape_img_data(data):
    for i in range(0,len(data)):
        data[i][0] = np.array(data[i][0])
        data[i][0] = data[i][0].reshape((IMG_SIZE*IMG_SIZE*3, 1))
    return data

def orig_img_data(data):
    for i in range(0, len(data)):
        data[i][0] = np.array(data[i][0])
        data[i][0] = data[i][0].reshape((IMG_SIZE, IMG_SIZE, 3))
    return data

def divide_X_Y(data):
    X = []
    Y = []
    for i in range(0, len(data)):
        X.append(data[i][0])
        Y.append(data[i][1])
    return X,Y
       
        
        
        
        
        