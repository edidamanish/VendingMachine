#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 01:58:28 2018

@author: manish
"""

import pandas as pd
import numpy as np
import random
import json

items = ['Pringles', 'Lays',	'Pepsi','Mirinda', 'Snickers','Diary Milk Chocolate','Cadbury 5 Star','Sprite','Bingo Mad Angles','Kurkure Puffcorn']

users = 100

bought_list = []

for i in range(0,users):
    list1 = [0]*10
    indexs = random.sample(range(1,10),random.randint(1,5))
    for j in range(0, len(indexs)):
        list1[indexs[j]] = random.randint(1,10)
    bought_list.append(list1)
    
    
dummy_dict = []
for i in range(0, users):
    prod = {}
    for j in range(0, 10):
        prod[items[j]] = bought_list[i][j]
    dummy_dict.append({'user' : 'user'+str(i+1), 'products' : prod})
    
with open('recomm_dummy.txt', 'w') as file1:
    json.dump(dummy_dict[:96], file1)

#with open('dummy_user_info.txt', 'w') as file1:
#    json.dump(dummy_dict[96:], file1)
dummy_dict2 = dummy_dict[96:]
machine_info = []
for line in open('vendingmachines.json', 'r'):
    machine_info.append(json.loads(line))

mach = []
for i in range(0, 4):
    avail_prods = []
    for j in range(0, len(machine_info[i]['products'])):
        avail_prods.append(machine_info[i]['products'][j]['Name'])
    mach.append({'vendId' : machine_info[i]['vendId'], 'availProds' : avail_prods})
    dummy_dict2[i]['vendId'] = machine_info[i]['vendId']
    dummy_dict2[i]['availProds'] = avail_prods
    
    
with open('recomm_and2scrip.txt', 'w') as file1:
    json.dump(dummy_dict2, file1)

