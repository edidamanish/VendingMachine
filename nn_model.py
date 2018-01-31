#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 21:59:21 2018

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

fold_list = os.listdir('/home/manish/Manish/Hackathon/google-images-download-master/Sample_Images')

train_data = np.load('train_data.npy')
test_data = np.load('test_data.npy')


m = len(train_data)
(n_H0, n_W0) = train_data[0][0].shape
n_y = train_data[0][1].shape[0]
train_shape = (m , n_H0, n_W0, 1)


train = train_data[:500]
test = train_data[500:]

X_train, Y_train = nn_utils.divide_X_Y(train)
X_test, Y_test = nn_utils.divide_X_Y(test)

X_train, X_test = np.array(X_train), np.array(X_test)
Y_train, Y_test = np.array(Y_train), np.array(Y_test)
(m1, n_H1, n_W1) = X_train.shape
X_train = X_train.reshape(m1, n_H1, n_W1, 1)
(m2, n_H2, n_W2) = X_test.shape
X_test= X_test.reshape(m2, n_H2, n_W2, 1)
#Y_train, Y_test = Y_train.T[0], Y_test.T[0]
#Y_train, Y_test = Y_train.T, Y_test.T

MODEL_NAME = 'LeNet_5'
IMG_SIZE = 100

import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d,avg_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')
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


model.save(MODEL_NAME)

#model.load(MODEL_NAME)

def gen_random_no(n_Low, n_High, num):
    list_rand = []
    for x in range(0,num):
        list_rand.append(random.randint(n_Low, n_High))
    return list_rand


def plot_test_images(data):
    m = data.shape[0]
    list_rand = gen_random_no(0,m-1,12)
    fig = plt.figure()
    for i in range(0,12):
        img = data[list_rand[i]]
        model_out = model.predict([img])[0]
        label = np.argmax(model_out)
        y = fig.add_subplot(3,4,i+1)
        img = img.reshape((100,100))
        y.imshow(img, cmap='Greys')
        plt.title(fold_list[int(label)])
        y.axes.get_xaxis().set_visible(False)
        y.axes.get_yaxis().set_visible(False)
    plt.show()


plot_test_images(X_test)

#def check_accuracy(X_train, Y_train, X_test, Y_test):
#    m = X_train.shape[0]
#    n = X_test.shape[0]
#    label_pred = []
#    label_orig = []
#    for i in range(0,m):
#        img = X_train[i]
#        label_pred.append(np.argmax(model.predict([img])[0]))
#        label_orig.append(np.argmax(Y_train[i],axis=0))
#    label_pred, label_orig = np.array(label_pred), np.array(label_orig)
#    train_accuracy = (label_pred == label_orig).all(axis=0).mean()
#    label_pred = []
#    label_orig = []   
#    for i in range(0,n):
#        img = X_test[i]
#        label_pred.append(np.argmax(model.predict([img])[0]))
#        label_orig.append(np.argmax(Y_test[i],axis=0))   
#    label_pred, label_orig = np.array(label_pred), np.array(label_orig)
#    test_accuracy = (label_pred == label_orig).all(axis=0).mean()
#    return train_accuracy, test_accuracy
    

#train_accuracy, test_accuracy = check_accuracy(X_train, Y_train, X_test, Y_test)
#print("Train accuracy: "+str(train_accuracy))
#print("Test accuracy: " +str(test_accuracy))