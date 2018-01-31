#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 01:53:48 2018

@author: manish
"""

#u = [1,0,3,0,0,4,0,0,0,1]

import json
import numpy as np
with open('recomm_dummy.txt', 'rb') as file1:
    machine_info = json.load(file1)
    

    
items = ['Pringles', 'Lays',	'Pepsi','Mirinda','Snickers',
         'Diary Milk Chocolate','Cadbury 5 Star','Sprite',
         'Bingo Mad Angles','Kurkure Puffcorn']

users = len(machine_info)

user_his = []
for i in range(0, len(machine_info)):
    list1 = [0]*10
    for j in range(0, 10):
        dict1 = machine_info[i]['products']
        list1[j] = dict1[items[j]] 
    user1 = machine_info[i]['user']
    user_his.append([[user1],list1])
    
def unit_vector(vector):
    return vector / np.linalg.norm(vector)

def angle(v1, v2):

    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

recomm2 = user_his

sim = []

for i in range(0, len(recomm2)):
    if max(recomm2[i][1]) != 0:
        sim.append([recomm2[i][0][0], angle(u, recomm2[i][1]), recomm2[i][1]])
    
sim = sorted(sim, key = lambda x:(x[1]))

sim_user = sim[0:5]

recomm_prod = []
recomm_prod_itm = []
for i in range(0, len(sim_user)):
    j = 0 
    while(j<3):
        j+=1
        recomm_prod.append([items[np.argmax(sim_user[i][2])], sim_user[i][2][np.argmax(sim_user[i][2])] ])
        recomm_prod_itm.append(items[np.argmax(sim_user[i][2])])
        sim_user[i][2][np.argmax(sim_user[i][2])] = 0
        
recomm_prod_itm = set(recomm_prod_itm)
recomm_prod_itm = list(recomm_prod_itm)


recomm_prod.sort()
recomm_prod_upd = []

for item in recomm_prod_itm:
    count = 0
    for i in range(0, len(recomm_prod)):
        if item == recomm_prod[i][0]:
            count+= recomm_prod[i][1]
    recomm_prod_upd.append([item, count])
    
recomm_prod_upd = sorted(recomm_prod_upd, key = lambda x:(x[1]))

recommendations = []

for i in range(0, 5):
    recommendations.append(recomm_prod_upd[len(recomm_prod_upd) - i -1][0])

print(recommendations)

def convert_dict_2_list(dict2):
    user_his1 = []
    list1 = [0]*10
    for j in range(0, 10):
        dict1 = dict['products']
        list1[j] = dict1[items[j]] 
    user1 = machine_info['user']
    user_his1.append([[user1],list1])
    return user_his1
