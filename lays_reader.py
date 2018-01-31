#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 20:35:49 2018

@author: manish
"""



import numpy as np
import math
import random
import matplotlib.pyplot as plt
from random import shuffle
import os
import cv2
import tqdm

label_grid = ["Pringles",	"Lays",	"Pepsi","Mirinda","Snickers",	"Dairy Milk", "5Star","Sprite","Mad Angles","Puffcorn"]

labels_one_hot = np.zeros((len(label_grid),len(label_grid)), dtype = int)
j=0
for i in range(0, len(labels_one_hot)):
   labels_one_hot[i][j] = 1
   j+=1     

TRAIN_DIR = '/home/manish/Manish/Hackathon/google-images-download-master/Sample_Images'
fold_list = os.listdir('/home/manish/Manish/Hackathon/google-images-download-master/Sample_Images')

def create_train_test_set():
    training_data = []
    for i in range(0, len(fold_list)):
        IMG_DIR = os.path.join(TRAIN_DIR, fold_list[i])
        j = 0
        img_list = os.listdir(IMG_DIR)
        for img in img_list[0:70]:
            print(j)
            j+=1
            label_index = fold_list.index(fold_list[i])
            label = labels_one_hot[label_index]
            path = os.path.join(IMG_DIR, img)
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is not None:
                img = cv2.resize(img, (100,100))
                training_data.append([np.array(img), np.array(label)])
    shuffle(training_data)  
    np.save('train_data.npy', training_data)
    
    testing_data = []
    for i in range(0, len(fold_list)):
        IMG_DIR = os.path.join(TRAIN_DIR, fold_list[i])
        j = 0
        img_list = os.listdir(IMG_DIR)
        for img in img_list[70:]:
            print(j)
            j+=1
            label_index = fold_list.index(fold_list[i])
            label = labels_one_hot[label_index]
            path = os.path.join(IMG_DIR, img)
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is not None:
                img = cv2.resize(img, (100,100))
                testing_data.append([np.array(img), np.array(label)])
    shuffle(testing_data)  
    np.save('test_data.npy', testing_data)
    
    return training_data, testing_data


#def create_train_test_set():
#    training_data = []
#    i=0
#    for img in image_list[0:80]:
#        print(i)
#        i+=1
#        label = labels_one_hot[1]
#        path = os.path.join(TRAIN_DIR, img)
#        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
#        img = cv2.resize(img, (100,100))
#        training_data.append([np.array(img), np.array(label)])
#    shuffle(training_data)  
#    np.save('lays_train.npy', training_data)
#    
#    testing_data = []
#    i = 0
#    for img in image_list[80:]:
#        print(i)
#        i+=1
#        label = labels_one_hot[1]
#        path = os.path.join(TRAIN_DIR, img)
#        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
#        img = cv2.resize(img, (100,100))
#        testing_data.append([np.array(img), np.array(label)])
#    shuffle(testing_data)  
#    np.save('lays_test.npy', testing_data)
#    
#    return training_data, testing_data  
    

train_data, test_data = create_train_test_set()
