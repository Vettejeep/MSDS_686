# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 21:31:49 2018
@author: Kevin Maher
"""
# early pooor performing model for the project
# it got a lot better with time and experience!

import os
import numpy as np
import matplotlib.pyplot as plt

from keras.applications import Xception
from keras.preprocessing.image import ImageDataGenerator
from keras import models
from keras import layers
from keras import optimizers
from keras import backend as K

import tensorflow as tf

K.set_floatx('float32')

base_dir = r'D:\$Plants'

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

batch_size = 20
seed = 321
im_size = 240  # memory size limitations may require use of a smaller image than the network was designed for :(

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
                                                        batch_size=25,
                                                        class_mode='categorical')

validation_generator = validation_datagen.flow_from_directory(validation_dir,
                                                                target_size=(im_size, im_size),
                                                                batch_size=25,
                                                                class_mode='categorical')

test_generator = test_datagen.flow_from_directory(test_dir,
                                                        target_size=(im_size, im_size),
                                                        batch_size=25,
                                                        class_mode='categorical')

conv_base = Xception(weights='imagenet',
                              include_top=False,
                              input_shape=(im_size, im_size, 3))
# conv_base.summary()
conv_base.trainable = False


def build_model():
    model = models.Sequential()
    model.add(conv_base)
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(12, activation='softmax')) 
    model.compile(loss='categorical_crossentropy',
                      optimizer=optimizers.RMSprop(lr=1e-5),
                      metrics=['acc'])
    return model


with tf.device('/gpu:0'):
    np.random.seed(seed)
    model = build_model()
    print('Pre-train dense layers')
    history = model.fit_generator(train_generator,
                                  steps_per_epoch=114,
                                  epochs=6,
                                  validation_data=validation_generator,
                                  validation_steps=48,
                                  verbose=1,
                                  workers=8)

    conv_base.trainable = True

    set_trainable = False
    for layer in conv_base.layers:
        if 'block13' in layer.name:
            set_trainable = True
            print('Set Trainable')
        if set_trainable:
            layer.trainable = True
        else:
            layer.trainable = False

    print('Train Model to block 13')
    np.random.seed(seed)
    model = build_model()
    history = model.fit_generator(train_generator,
                                  steps_per_epoch=228,
                                  epochs=12,
                                  validation_data=validation_generator,
                                  validation_steps=48,
                                  verbose=1,
                                  initial_epoch=6,
                                  workers=8)


    set_trainable = False
    for layer in conv_base.layers:
        if 'block11' in layer.name:
            set_trainable = True
            print('Set Trainable')
        if set_trainable:
            layer.trainable = True
        else:
            layer.trainable = False

    print('Train Model to block 11')
    np.random.seed(seed)
    model = build_model()
    history = model.fit_generator(train_generator,
                                  steps_per_epoch=228,
                                  epochs=18,
                                  validation_data=validation_generator,
                                  validation_steps=48,
                                  verbose=1,
                                  initial_epoch=12,
                                  workers=8)
    
    scores = model.evaluate_generator(test_generator, workers=8)
    print('#1 Loss, Accuracy: ', scores)


print('Plot Model')
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

#Train Model to block 11
#Epoch 13/18
#228/228 [==============================] - 57s 250ms/step - loss: 1.1975 - acc: 0.6148 - val_loss: 1.0161 - val_acc: 0.6428
#Epoch 14/18
#228/228 [==============================] - 46s 201ms/step - loss: 0.6630 - acc: 0.7779 - val_loss: 0.8887 - val_acc: 0.6786
#Epoch 15/18
#228/228 [==============================] - 45s 199ms/step - loss: 0.5531 - acc: 0.8104 - val_loss: 0.8274 - val_acc: 0.7176
#Epoch 16/18
#228/228 [==============================] - 46s 200ms/step - loss: 0.4895 - acc: 0.8279 - val_loss: 0.7463 - val_acc: 0.7250
#Epoch 17/18
#228/228 [==============================] - 46s 200ms/step - loss: 0.4356 - acc: 0.8518 - val_loss: 0.7740 - val_acc: 0.7429
#Epoch 18/18
#228/228 [==============================] - 46s 200ms/step - loss: 0.3767 - acc: 0.8716 - val_loss: 0.7350 - val_acc: 0.7524
#1 Loss, Accuracy:  [0.7050180517479965, 0.7520920485881581]