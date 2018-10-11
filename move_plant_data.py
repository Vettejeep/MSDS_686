# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 21:07:37 2018

@author: Vette
"""
import os
import shutil
import random

plant_names =['Black-grass', 'Charlock', 'Cleavers', 'Common Chickweed', 'Common wheat', 'Fat Hen',
              'Loose Silky-bent', 'Maize', 'Scentless Mayweed', 'Shepherds Purse', 'Small-flowered Cranesbill', 'Sugar beet' ]

base = r'D:\$Plants'
train = 'train'
validation = 'validation'
test = 'test'
src = 'train_orig'

# run once to make the directories
#for n in plant_names:
#    p = os.path.join(base, train, n)
#    os.mkdir(p)
#    
#    p = os.path.join(base, validation, n)
#    os.mkdir(p)
#    
#    p = os.path.join(base, test, n)
#    os.mkdir(p)
    
# copy the plant files
for n in plant_names:
    p = os.path.join(base, src, n)
    files = os. listdir(p)
    random.shuffle(files)
    
    for fname in files[0:int(len(files)*.6)]:
        f_src = os.path.join(base, src, n, fname)
        f_dst = os.path.join(base, train, n, fname)
        shutil.copy(f_src, f_dst)

    for fname in files[int(len(files)*.6):int(len(files)*.8)]:
        f_src = os.path.join(base, src, n, fname)
        f_dst = os.path.join(base, validation, n, fname)
        shutil.copy(f_src, f_dst)    

    for fname in files[int(len(files)*.8):]:
        f_src = os.path.join(base, src, n, fname)
        f_dst = os.path.join(base, test, n, fname)
        shutil.copy(f_src, f_dst)


print('Done')

