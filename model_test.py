#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 00:49:05 2018

@author: manish
"""

import numpy as np
import math
import random
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import cv2
import tqdm

import nn_utils
from cnn_utils import *

IMG_SIZE = 100
fold_list = os.listdir('/home/manish/Manish/Hackathon/google-images-download-master/Sample_Images')
print(fold_list)
img = cv2.imread('/home/manish/Manish/Hackathon/google-images-download-master/dairy.jpg', cv2.IMREAD_COLOR)
img = cv2.resize(img,(IMG_SIZE, IMG_SIZE))
plt.imshow(img, cmap='Greys')
plt.show
img = img.reshape(IMG_SIZE, IMG_SIZE, 3)




MODEL_NAME = 'LeNet_5_COLOR'
IMG_SIZE = 100

import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d,avg_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3], name='input')
convnet = conv_2d(convnet, nb_filter = 6, filter_size = 5, strides = 1, activation = 'relu')
convnet = avg_pool_2d(convnet, kernel_size =2, strides= 2)
convnet = conv_2d(convnet, nb_filter = 16, filter_size = 5, strides = 1, activation = 'relu')
convnet = avg_pool_2d(convnet, kernel_size =2, strides= 2)
convnet = fully_connected(convnet, 120, activation='relu')
convnet = fully_connected(convnet, 84, activation='relu')
convnet = fully_connected(convnet, 10, activation='softmax')
convnet = regression(convnet, optimizer= 'adam', learning_rate = 0.001, loss= 'categorical_crossentropy',name='targets')

model = tflearn.DNN(convnet, tensorboard_dir = 'log',tensorboard_verbose=3)

#model.fit({'input':X_train}, {'targets' : Y_train}, n_epoch = 10, validation_set= ({'input':X_test},{'targets':Y_test}), snapshot_step =500, show_metric = True, run_id= MODEL_NAME)


if os.path.exists('/home/manish/Manish/Hackathon/google-images-download-master/{}.meta'.format(MODEL_NAME)):
    model.load(MODEL_NAME)
    print('Model Loaded')
   
model_out = model.predict([img])[0]
print(model_out)
label = np.argmax(model_out)
print(fold_list[label])
print(label)
