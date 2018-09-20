# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 21:31:49 2018

@author: Vette
"""

import os
import math
import datetime
import gc

import numpy as np
import pandas as pd

from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import layers
from tensorflow.python.keras.applications.imagenet_utils import _obtain_input_shape
from tensorflow.python.keras.layers import Activation
from tensorflow.python.keras.layers import BatchNormalization
from tensorflow.python.keras.layers import Conv2D
from tensorflow.python.keras.layers import GlobalAveragePooling2D
from tensorflow.python.keras.layers import GlobalMaxPooling2D
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.layers import MaxPooling2D
from tensorflow.python.keras.layers import SeparableConv2D
from tensorflow.python.keras.layers import Flatten
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.models import Model
from tensorflow.python.keras import optimizers

import tensorflow as tf

K.set_floatx('float32')

base_dir = r'D:\$Plants'

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

EPOCHS = 24
train_size= 2845
validation_size = 949
test_size = 956
batch_size = 20
seed = 321
im_size = 224
num_classes = 12

base_train_steps = math.ceil(train_size / batch_size)
base_val_steps = math.ceil(validation_size / batch_size)
base_test_steps = math.ceil(test_size / batch_size)
print(base_train_steps, base_val_steps, base_test_steps)


train_datagen = ImageDataGenerator(rescale=1./255,
                                      rotation_range=45,
                                      width_shift_range=0.2,
                                      height_shift_range=0.2,
                                      shear_range=0.2,
                                      zoom_range=0.2,
                                      horizontal_flip=True,
                                      vertical_flip=True,
                                      fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir,
                                                        target_size=(im_size, im_size),
                                                        batch_size=batch_size,
                                                        class_mode='categorical')

validation_generator = validation_datagen.flow_from_directory(validation_dir,
                                                                target_size=(im_size, im_size),
                                                                batch_size=batch_size,
                                                                class_mode='categorical')

test_generator = test_datagen.flow_from_directory(test_dir,
                                                        target_size=(im_size, im_size),
                                                        batch_size=batch_size,
                                                        class_mode='categorical')


def build_model(input_shape=None, fdl_nodes=128, sdl_nodes=0, use_dense_dropout=False, 
                learning_rate=0.001, decay=0.0, pooling='max'):

    
    input_shape = _obtain_input_shape(
        input_shape,
        default_size=im_size,
        min_size=24,
        data_format=K.image_data_format(),
        require_flatten=False,
        weights='None')

    img_input = Input(shape=input_shape)

    # first block
    x = Conv2D(32, (3, 3), strides=(2, 2), use_bias=False, name='block1_conv1')(img_input)
    x = BatchNormalization(name='block1_conv1_bn')(x)
    x = Activation('relu', name='block1_conv1_act')(x)
    x = Conv2D(64, (3, 3), use_bias=False, name='block1_conv2')(x)
    x = BatchNormalization(name='block1_conv2_bn')(x)
    x = Activation('relu', name='block1_conv2_act')(x)

    residual = Conv2D(128, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    # second block
    x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False, name='block2_sepconv1')(x)
    x = BatchNormalization(name='block2_sepconv1_bn')(x)
    x = Activation('relu', name='block2_sepconv2_act')(x)
    x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False, name='block2_sepconv2')(x)
    x = BatchNormalization(name='block2_sepconv2_bn')(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block2_pool')(x)
    x = layers.add([x, residual])

    residual = Conv2D(512, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    # third block
    x = Activation('relu', name='block3_sepconv1_act')(x)
    x = SeparableConv2D(512, (3, 3), padding='same', use_bias=False, name='block3_sepconv1')(x)
    x = BatchNormalization(name='block3_sepconv1_bn')(x)
    x = Activation('relu', name='block3_sepconv2_act')(x)
    x = SeparableConv2D(512, (3, 3), padding='same', use_bias=False, name='block3_sepconv2')(x)
    x = BatchNormalization(name='block3_sepconv2_bn')(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block3_pool')(x)
    x = layers.add([x, residual])

    for i in range(5):
        residual = x
        prefix = 'block' + str(i + 4)

        x = Activation('relu', name=prefix + '_sepconv1_act')(x)
        x = SeparableConv2D(
            512, (3, 3), padding='same', use_bias=False, name=prefix + '_sepconv1')(x)
        x = BatchNormalization(name=prefix + '_sepconv1_bn')(x)
        x = Activation('relu', name=prefix + '_sepconv2_act')(x)
        x = SeparableConv2D(512, (3, 3), padding='same', use_bias=False, name=prefix + '_sepconv2')(x)
        x = BatchNormalization(name=prefix + '_sepconv2_bn')(x)
        x = Activation('relu', name=prefix + '_sepconv3_act')(x)
        x = SeparableConv2D(
            512, (3, 3), padding='same', use_bias=False, name=prefix + '_sepconv3')(x)
        x = BatchNormalization(name=prefix + '_sepconv3_bn')(x)

        x = layers.add([x, residual])

    residual = Conv2D(768, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    # output blocks - block 21  
    x = Activation('relu', name='block21_sepconv1_act')(x)
    x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name='block21_sepconv1')(x)
    x = BatchNormalization(name='block21_sepconv1_bn')(x)
    x = Activation('relu', name='block21_sepconv2_act')(x)
    x = SeparableConv2D(768, (3, 3), padding='same', use_bias=False, name='block21_sepconv2')(x)
    x = BatchNormalization(name='block21_sepconv2_bn')(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block13_pool')(x)
    x = layers.add([x, residual])

    # block 22
    x = SeparableConv2D(1024, (3, 3), padding='same', use_bias=False, name='block22_sepconv1')(x)
    x = BatchNormalization(name='block22_sepconv1_bn')(x)
    x = Activation('relu', name='block22_sepconv1_act')(x)

    x = SeparableConv2D(1536, (3, 3), padding='same', use_bias=False, name='block22_sepconv2')(x)
    x = BatchNormalization(name='block22_sepconv2_bn')(x)
    x = Activation('relu', name='block22_sepconv2_act')(x)

    if pooling == 'avg':
        x = GlobalAveragePooling2D()(x)
    elif pooling == 'max':
        x = GlobalMaxPooling2D()(x)

    x = Flatten()(x)
    x = Dense(fdl_nodes, activation='relu')(x)

    if use_dense_dropout:
        x = Dropout(rate=0.2)(x)

    if sdl_nodes > 0:
        x = Dense(sdl_nodes, activation='relu')(x)

    x = Dense(num_classes, activation='softmax')(x)

    model = Model(img_input, x, name='mini_xception_v1')
    opt = optimizers.Adam(lr=learning_rate, decay=decay)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# factors for a 4 level box-behnken design


doe = [[-1., -1.,  0.,  0.],
       [ 1., -1.,  0.,  0.],
       [-1.,  1.,  0.,  0.],
       [ 1.,  1.,  0.,  0.],
       [-1.,  0., -1.,  0.],
       [ 1.,  0., -1.,  0.],
       [-1.,  0.,  1.,  0.],
       [ 1.,  0.,  1.,  0.],
       [-1.,  0.,  0., -1.],
       [ 1.,  0.,  0., -1.],
       [-1.,  0.,  0.,  1.],
       [ 1.,  0.,  0.,  1.],
       [ 0., -1., -1.,  0.],
       [ 0.,  1., -1.,  0.],
       [ 0., -1.,  1.,  0.],
       [ 0.,  1.,  1.,  0.],
       [ 0., -1.,  0., -1.],
       [ 0.,  1.,  0., -1.],
       [ 0., -1.,  0.,  1.],
       [ 0.,  1.,  0.,  1.],
       [ 0.,  0., -1., -1.],
       [ 0.,  0.,  1., -1.],
       [ 0.,  0., -1.,  1.],
       [ 0.,  0.,  1.,  1.],
       [ 0.,  0.,  0.,  0.]]

# hyper-parameters or factors
lr_def = (0.00075, 0.00025)
fl_def = (256, 128)
sl_def = (48, 16)
decay_def = (0.0005, 0.0005)  # no momentum for adam

df = None
i = 1

with tf.device('/gpu:0'):
    for w, x, y, z in doe:
        lt0 = datetime.datetime.now()
        lr = lr_def[0] + (lr_def[1] * w)
        fl = int(fl_def[0] + (fl_def[1] * x))
        sl = int(sl_def[0] + (sl_def[1] * y))
        d = decay_def[0] + (decay_def[1] * z)
    
        np.random.seed(seed)
        print('Learning Rate: %.6f, FL Nodes: %d, SL Nodes: %d, decay: %.6f, iter: %d' % (lr, fl, sl, d, i))
        model = build_model(input_shape=(im_size, im_size, 3), fdl_nodes=fl, sdl_nodes=sl, learning_rate=lr, decay=d)
        print('Train')
        history = model.fit_generator(train_generator,
                                      steps_per_epoch=base_train_steps * 2,
                                      epochs=EPOCHS,
                                      validation_data=validation_generator,
                                      validation_steps=base_val_steps,
                                      verbose=1,
                                      workers=8)
    
        scores = model.evaluate_generator(test_generator, steps=base_test_steps, workers=8)
        print('#1 Loss, Accuracy: ', scores)
              
        accs = history.history['acc']
        val_accs = history.history['val_acc']
        losses = history.history['loss']
        val_losses = history.history['val_loss']
        
        data = {'iter': [i for j in range(EPOCHS)], 'epoch': [j+1 for j in range(EPOCHS)],
                'lr_code': [w for j in range(EPOCHS)], 'fl_code': [x for j in range(EPOCHS)],
                'sl_code': [y for j in range(EPOCHS)], 'decay_code': [z for j in range(EPOCHS)],
                'learning_rate': [lr for j in range(EPOCHS)], 'fl_nodes': [fl for j in range(EPOCHS)],
                'sl_nodes': [sl for j in range(EPOCHS)], 'decay': [d for j in range(EPOCHS)],
                'acc': accs, 'loss': losses, 'val_acc': val_accs, 'val_loss': val_losses,
                'test_final_loss': [scores[0]] * EPOCHS, 'test_final_acc': [scores[1]] * EPOCHS}

        print(data)
                    
        if df is None:
            df = pd.DataFrame(data=data, index=None)
        else:
            temp = pd.DataFrame(data=data, index=None)
            df = df.append(temp, ignore_index=True)
           
        df = df[['iter', 'epoch', 'lr_code', 'fl_code', 'sl_code', 'decay_code', 
                 'learning_rate', 'fl_nodes', 'sl_nodes', 'decay',
                 'acc', 'loss', 'val_acc','val_loss', 'test_final_loss', 'test_final_acc']] 
        df.reset_index(inplace=True, drop=True)
        print('Best:')
        dfs = df.sort_values(by=['val_acc', 'acc'], ascending=[False, False], inplace=False)    
        print(dfs.head(3))
        df.to_csv('mini_xp_doe_{}.csv'.format(i), index=False)
        
        lt = round((datetime.datetime.now() - lt0).total_seconds(), 2)
        print('Loop Time: {0}'.format(lt))
        i += 1
        
        try:
            K.clear_session()  # https://github.com/keras-team/keras/issues/2102
        except:
            print('EXCEPTION in model cleanup!')
