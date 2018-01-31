#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 17:38:39 2018

@author: manish
"""

import pandas as pd
import numpy as np

recomm = pd.read_csv('recomm-dummy.csv')

recomm1 = recomm.as_matrix()
col_head = recomm.columns
col_head = list(col_head)
col_head.remove(col_head[0])
favorites = {}

users, items = recomm1.shape
items-=1

    
def unit_vector(vector):
    return vector / np.linalg.norm(vector)

def angle(v1, v2):

    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def split(list1 , index=1):
    x, y = [] , []
    for i in range(0, len(list1)):
        if i <= index-1:
            x.append(list1[i])
        else:
            y.append(list1[i])
    return [x, y]

recomm2 = []

for i in range(0, users):
    recomm2.append(split(recomm1[i]))
    
u = [1,0,3,0,0,4,0,0,0,1]

sim = []

for i in range(0, len(recomm2)):
    sim.append([recomm2[i][0][0], angle(u, recomm2[i][1]), recomm2[i][1]])
    
sim = sorted(sim, key = lambda x:(x[1]))

sim_user = sim[0:5]

recomm_prod = []
recomm_prod_itm = []
for i in range(0, len(sim_user)):
    j = 0 
    while(j<3):
        j+=1
        recomm_prod.append([col_head[np.argmax(sim_user[i][2])], sim_user[i][2][np.argmax(sim_user[i][2])] ])
        recomm_prod_itm.append(col_head[np.argmax(sim_user[i][2])])
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
    











    



