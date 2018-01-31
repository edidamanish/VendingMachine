#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 13:29:09 2018

@author: manish
"""

import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import urllib
import json
import random

file1 = open('items.txt' , 'r')
items = file1.read()
items = items.split(', ')
items.remove(items[-1])

ID = 100000
LONG_MIN = 76.3
LONG_MAX = 77.2
LAT_MIN = 27.9
LAT_MAX = 28.7

product_info = []

product_info_dict = {}

r = urllib.request.urlopen('http://www.myfitnesspal.com/food/calories/pringles-spicy-chips-436792675').read()
soup = BeautifulSoup(r)

table_tag=soup.table
table_txt = table_tag.text
table_contents = table_txt.split('\n')
table_contents = [x for x in table_contents if x != '' and x!= '\xa0']
content_dict = {}

i = 0
while(i<len(table_contents)):
    content_dict[table_contents[i]] = table_contents[i+1]
    i+=2

product_info_dict['ID']= ID
product_info_dict['Name'] = 'Pringles'
product_info_dict['Nutrition Facts'] = content_dict
product_info_dict['Maximum Quantity'] = 20
product_info_dict['Cur Quantity'] = 10
product_info_dict['Image'] = 'https://www.dollartree.com/assets/product_images_2016/styles/xlarge/198606.jpg'

ID+=1


product_info.append(product_info_dict)

product_info_dict2 = {}

r = urllib.request.urlopen('http://www.myfitnesspal.com/food/calories/350783671').read()
soup = BeautifulSoup(r)

table_tag=soup.table
table_txt = table_tag.text
table_contents = table_txt.split('\n')
table_contents = [x for x in table_contents if x != '' and x!= '\xa0']
content_dict = {}

i = 0
while(i<len(table_contents)):
    content_dict[table_contents[i]] = table_contents[i+1]
    i+=2
    
product_info_dict2['ID']= ID
product_info_dict2['Name'] = 'Lays'
product_info_dict2['Nutrition Facts'] = content_dict
product_info_dict2['Maximum Quantity'] = 30
product_info_dict2['Cur Quantity'] = 15
product_info_dict2['Image'] = 'http://origin-www.fritolay.com/images/default-source/blue-bag-image/lays-classic.png?sfvrsn=bd1e563a_2'

ID+=1

product_info.append(product_info_dict2)


#################################################################
product_info_dict2 = {}

r = urllib.request.urlopen('http://www.myfitnesspal.com/food/calories/205691249').read()
soup = BeautifulSoup(r)

table_tag=soup.table
table_txt = table_tag.text
table_contents = table_txt.split('\n')
table_contents = [x for x in table_contents if x != '' and x!= '\xa0']
content_dict = {}

i = 0
while(i<len(table_contents)):
    content_dict[table_contents[i]] = table_contents[i+1]
    i+=2
    
product_info_dict2['ID']= ID
product_info_dict2['Name'] = 'Snickers'
product_info_dict2['Nutrition Facts'] = content_dict
product_info_dict2['Maximum Quantity'] = 50
product_info_dict2['Cur Quantity'] = 35
product_info_dict2['Image'] = 'https://images-na.ssl-images-amazon.com/images/I/71%2Br1gAwsZL._SX466_.jpg'
ID+=1

product_info.append(product_info_dict2)

#################################################################
product_info_dict2 = {}

r = urllib.request.urlopen('http://www.myfitnesspal.com/food/calories/495985960').read()
soup = BeautifulSoup(r)

table_tag=soup.table
table_txt = table_tag.text
table_contents = table_txt.split('\n')
table_contents = [x for x in table_contents if x != '' and x!= '\xa0']
content_dict = {}

i = 0
while(i<len(table_contents)):
    content_dict[table_contents[i]] = table_contents[i+1]
    i+=2
    
product_info_dict2['ID']= ID
product_info_dict2['Name'] = 'Mirinda'
product_info_dict2['Nutrition Facts'] = content_dict
product_info_dict2['Maximum Quantity'] = 35
product_info_dict2['Cur Quantity'] = 6
product_info_dict2['Image'] = 'https://4.imimg.com/data4/CC/CC/GLADMIN-/wp-content-uploads-2016-04-mirinda-can_2-1-500x500.jpg'
ID+=1

product_info.append(product_info_dict2)

#################################################################
product_info_dict2 = {}

r = urllib.request.urlopen('http://www.myfitnesspal.com/food/calories/313182234').read()
soup = BeautifulSoup(r)

table_tag=soup.table
table_txt = table_tag.text
table_contents = table_txt.split('\n')
table_contents = [x for x in table_contents if x != '' and x!= '\xa0']
content_dict = {}

i = 0
while(i<len(table_contents)):
    content_dict[table_contents[i]] = table_contents[i+1]
    i+=2
    
product_info_dict2['ID']= ID
product_info_dict2['Name'] = 'Pepsi'
product_info_dict2['Nutrition Facts'] = content_dict
product_info_dict2['Maximum Quantity'] = 25
product_info_dict2['Cur Quantity'] = 10
product_info_dict2['Image'] = 'https://images-na.ssl-images-amazon.com/images/I/61kFbWto%2BOL._SY355_.jpg'
ID+=1

product_info.append(product_info_dict2)

#################################################################
product_info_dict2 = {}

r = urllib.request.urlopen('http://www.myfitnesspal.com/food/calories/268142290').read()
soup = BeautifulSoup(r)

table_tag=soup.table
table_txt = table_tag.text
table_contents = table_txt.split('\n')
table_contents = [x for x in table_contents if x != '' and x!= '\xa0']
content_dict = {}

i = 0
while(i<len(table_contents)):
    content_dict[table_contents[i]] = table_contents[i+1]
    i+=2
    
product_info_dict2['ID']= ID
product_info_dict2['Name'] = 'Diary Milk Chocolate'
product_info_dict2['Nutrition Facts'] = content_dict
product_info_dict2['Maximum Quantity'] = 40
product_info_dict2['Cur Quantity'] = 30
product_info_dict2['Image'] = 'https://5.imimg.com/data5/CM/GU/MY-44229728/dairy-milk-chocolate-500x500.jpg'
ID+=1

product_info.append(product_info_dict2)

#################################################################
product_info_dict2 = {}

r = urllib.request.urlopen('http://www.myfitnesspal.com/food/calories/474849318').read()
soup = BeautifulSoup(r)

table_tag=soup.table
table_txt = table_tag.text
table_contents = table_txt.split('\n')
table_contents = [x for x in table_contents if x != '' and x!= '\xa0']
content_dict = {}

i = 0
while(i<len(table_contents)):
    content_dict[table_contents[i]] = table_contents[i+1]
    i+=2
    
product_info_dict2['ID']= ID
product_info_dict2['Name'] = 'Kurkure Puffcorn'
product_info_dict2['Nutrition Facts'] = content_dict
product_info_dict2['Maximum Quantity'] = 25
product_info_dict2['Cur Quantity'] = 14
product_info_dict2['Image'] = 'https://images-na.ssl-images-amazon.com/images/I/81q5sKbW9WL._SX342_.jpg'
ID+=1

product_info.append(product_info_dict2)

#################################################################
product_info_dict2 = {}

r = urllib.request.urlopen('http://www.myfitnesspal.com/food/calories/536097994').read()
soup = BeautifulSoup(r)

table_tag=soup.table
table_txt = table_tag.text
table_contents = table_txt.split('\n')
table_contents = [x for x in table_contents if x != '' and x!= '\xa0']
content_dict = {}

i = 0
while(i<len(table_contents)):
    content_dict[table_contents[i]] = table_contents[i+1]
    i+=2
    
product_info_dict2['ID']= ID
product_info_dict2['Name'] = 'Sprite'
product_info_dict2['Nutrition Facts'] = content_dict
product_info_dict2['Maximum Quantity'] = 30
product_info_dict2['Cur Quantity'] = 23
product_info_dict2['Image'] = 'https://www.diabolopizza.ch/wp-content/uploads/2017/03/sprite.png'
ID+=1

product_info.append(product_info_dict2)

#################################################################
product_info_dict2 = {}

r = urllib.request.urlopen('http://www.myfitnesspal.com/food/calories/613075382').read()
soup = BeautifulSoup(r)

table_tag=soup.table
table_txt = table_tag.text
table_contents = table_txt.split('\n')
table_contents = [x for x in table_contents if x != '' and x!= '\xa0']
content_dict = {}

i = 0
while(i<len(table_contents)):
    content_dict[table_contents[i]] = table_contents[i+1]
    i+=2
    
product_info_dict2['ID']= ID
product_info_dict2['Name'] = 'Bingo Mad Angles'
product_info_dict2['Nutrition Facts'] = content_dict
product_info_dict2['Maximum Quantity'] = 40
product_info_dict2['Cur Quantity'] = 16
product_info_dict2['Image'] = 'http://toksale.com/wp-content/uploads/2016/03/Bingo-Mad-Angles-Achaari-Masti.jpg'
ID+=1

product_info.append(product_info_dict2)

#################################################################
product_info_dict2 = {}

r = urllib.request.urlopen('http://www.myfitnesspal.com/food/calories/583856053').read()
soup = BeautifulSoup(r)

table_tag=soup.table
table_txt = table_tag.text
table_contents = table_txt.split('\n')
table_contents = [x for x in table_contents if x != '' and x!= '\xa0']
content_dict = {}

i = 0
while(i<len(table_contents)):
    content_dict[table_contents[i]] = table_contents[i+1]
    i+=2
    
product_info_dict2['ID']= ID
product_info_dict2['Name'] = 'Cadbury 5 Star'
product_info_dict2['Nutrition Facts'] = content_dict
product_info_dict2['Maximum Quantity'] = 50
product_info_dict2['Cur Quantity'] = 34
product_info_dict2['Image'] = 'https://images-na.ssl-images-amazon.com/images/I/31b0WlqeZOL.jpg'
ID+=1

product_info.append(product_info_dict2)
#################################################################

vm_info = []
VM_NO = 5
PROD_NO = len(product_info)
VM_ID = 1
for i in range(0, VM_NO):
    p_inf = []
    a1 = np.random.randint(PROD_NO , size= random.randint(1, 8))
    a1 = set(a1)
    a1 = list(a1)
    for i in range(0, len(a1)):
        p_inf.append(product_info[a1[i]])
    vm_info.append({'vendId' : VM_ID, 'products': p_inf, 'locLat' : random.uniform(LAT_MIN, LAT_MAX), 
                    'locLang' : random.uniform(LONG_MIN, LONG_MAX)})

with open('machine_database.txt', 'w') as file1:
    json.dump(vm_info, file1)