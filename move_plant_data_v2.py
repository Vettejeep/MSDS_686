# -*- coding: utf-8 -*-
"""
Created on Oct 3, 2018

@author: Kevin Maher
"""
import os
import shutil
import random

random.seed(5591)
plant_names =['Black-grass', 'Charlock', 'Cleavers', 'Common Chickweed', 'Common wheat', 'Fat Hen',
              'Loose Silky-bent', 'Maize', 'Scentless Mayweed', 'Shepherds Purse', 'Small-flowered Cranesbill', 'Sugar beet' ]

base = r'D:\$Plants'
train = 'train70'
validation = 'validation30'
src = 'train_orig'

p = os.path.join(base, train)
os.mkdir(p)

p = os.path.join(base, validation)
os.mkdir(p)

# run once to make the directories
for n in plant_names:
    p = os.path.join(base, train, n)
    os.mkdir(p)
    
    p = os.path.join(base, validation, n)
    os.mkdir(p)
    
# copy the plant files
for n in plant_names:
    p = os.path.join(base, src, n)
    files = os. listdir(p)
    random.shuffle(files)
    
    for fname in files[0:int(len(files)*.7)]:
        f_src = os.path.join(base, src, n, fname)
        f_dst = os.path.join(base, train, n, fname)
        shutil.copy(f_src, f_dst)

    for fname in files[int(len(files)*.7):]:
        f_src = os.path.join(base, src, n, fname)
        f_dst = os.path.join(base, validation, n, fname)
        shutil.copy(f_src, f_dst)    

print('Done')
