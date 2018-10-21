# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 18:48:12 2018

@author: Vette
"""
import os
import math

import numpy as np
import pandas as pd

# from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import layers
from tensorflow.python.keras.applications.imagenet_utils import _obtain_input_shape
from tensorflow.python.keras.layers import Activation
from tensorflow.python.keras.layers import BatchNormalization
from tensorflow.python.keras.layers import Conv2D
from tensorflow.python.keras.layers import GlobalMaxPooling2D
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.layers import MaxPooling2D
from tensorflow.python.keras.layers import SeparableConv2D
from tensorflow.python.keras.layers import Flatten
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Model
from tensorflow.python.keras import optimizers
from tensorflow.python.keras import regularizers

import tensorflow as tf

K.clear_session() 
K.set_floatx('float32')

base_dir = r'C:\Users\Vette\Desktop\Regis\$Plants'
data_dir = r'data_mx_L2_submission'
data_path = os.path.join(base_dir, data_dir)
os.mkdir(data_path)

train_dir = os.path.join(base_dir, 'train_orig')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test_submission')

EPOCHS = 120
REPEATS = 10
train_size= 4750
validation_size = 949
test_size = 794
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

train_generator = train_datagen.flow_from_directory(train_dir,
                                                        target_size=(im_size, im_size),
                                                        batch_size=batch_size,
                                                        class_mode='categorical')

validation_generator = validation_datagen.flow_from_directory(validation_dir,
                                                                target_size=(im_size, im_size),
                                                                batch_size=batch_size,
                                                                class_mode='categorical')



def build_model(input_shape=None):
    input_shape = _obtain_input_shape(
        input_shape,
        default_size=im_size,
        min_size=24,
        data_format=K.image_data_format(),
        require_flatten=False,
        weights='None')

    img_input = Input(shape=input_shape)
    reg = regularizers.l2(0.001)
    
    # first block
    x = Conv2D(32, (3, 3), strides=(2, 2), use_bias=False, name='block1_conv1', kernel_regularizer=reg)(img_input)
    x = BatchNormalization(name='block1_conv1_bn')(x)
    x = Activation('relu', name='block1_conv1_act')(x)
    
    x = Conv2D(64, (3, 3), use_bias=False, name='block1_conv2', kernel_regularizer=reg)(x)
    x = BatchNormalization(name='block1_conv2_bn')(x)
    x = Activation('relu', name='block1_conv2_act')(x)

    residual = Conv2D(256, (1, 1), strides=(2, 2), padding='same', use_bias=False, name='residual_conv2d_1', kernel_regularizer=reg)(x)
    residual = BatchNormalization()(residual) 
   
    # second block
    x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False, name='block2_sepconv1', kernel_regularizer=reg)(x)
    x = BatchNormalization(name='block2_sepconv1_bn')(x)
    x = Activation('relu', name='block2_sepconv2_act')(x)
    x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False, name='block2_sepconv2', kernel_regularizer=reg)(x)
    x = BatchNormalization(name='block2_sepconv2_bn')(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block2_pool')(x)
    x = layers.add([x, residual])

    # second residual
    residual = Conv2D(256, (1, 1), strides=(2, 2), padding='same', use_bias=False, name='residual_conv2d_2', kernel_regularizer=reg)(x)
    residual = BatchNormalization()(residual)

    for i in range(2):
        residual = x
        prefix = 'block' + str(i + 3)

        x = Activation('relu', name=prefix + '_sepconv1_act')(x)
        x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False, name=prefix + '_sepconv1', kernel_regularizer=reg)(x)
        x = BatchNormalization(name=prefix + '_sepconv1_bn')(x)
        x = Activation('relu', name=prefix + '_sepconv2_act')(x)
        x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False, name=prefix + '_sepconv2', kernel_regularizer=reg)(x)
        x = BatchNormalization(name=prefix + '_sepconv2_bn')(x)
        x = Activation('relu', name=prefix + '_sepconv3_act')(x)
        x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False, name=prefix + '_sepconv3', kernel_regularizer=reg)(x)
        x = BatchNormalization(name=prefix + '_sepconv3_bn')(x)

        x = layers.add([x, residual])

    residual = Conv2D(384, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    # output blocks - block 21  
    x = Activation('relu', name='block21_sepconv1_act')(x)
    x = SeparableConv2D(384, (3, 3), padding='same', use_bias=False, name='block21_sepconv1', kernel_regularizer=reg)(x)
    x = BatchNormalization(name='block21_sepconv1_bn')(x)
    x = Activation('relu', name='block21_sepconv2_act')(x)
    x = SeparableConv2D(384, (3, 3), padding='same', use_bias=False, name='block21_sepconv2', kernel_regularizer=reg)(x)
    x = BatchNormalization(name='block21_sepconv2_bn')(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block13_pool')(x)
    x = layers.add([x, residual])

    # block 22
    x = SeparableConv2D(512, (3, 3), padding='same', use_bias=False, name='block22_sepconv1', kernel_regularizer=reg)(x)
    x = BatchNormalization(name='block22_sepconv1_bn')(x)
    x = Activation('relu', name='block22_sepconv1_act')(x)

    x = SeparableConv2D(768, (3, 3), padding='same', use_bias=False, name='block22_sepconv2', kernel_regularizer=reg)(x)
    x = BatchNormalization(name='block22_sepconv2_bn')(x)
    x = Activation('relu', name='block22_sepconv2_act')(x)
    
    # model finish
    x = GlobalMaxPooling2D()(x)
    x = Flatten()(x)
    x = Dense(256, activation='relu', kernel_regularizer=reg)(x)
    x = Dense(64, activation='relu', kernel_regularizer=reg)(x)
    x = Dense(num_classes, activation='softmax')(x)
    
    model = Model(img_input, x, name='micro_xception_bn_v1')
    opt = optimizers.Adam(lr=0.0008, decay=0.001)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

model = build_model(input_shape=(im_size, im_size, 3))
#model.summary()

# https://www.kaggle.com/miklgr500/keras-simple-model-0-97103-best-public-score
INV_CLASS = {
    0: 'Black-grass',
    1: 'Charlock',
    2: 'Cleavers',
    3: 'Common Chickweed',
    4: 'Common wheat',
    5: 'Fat Hen',
    6: 'Loose Silky-bent',
    7: 'Maize',
    8: 'Scentless Mayweed',
    9: 'Shepherds Purse',
    10: 'Small-flowered Cranesbill',
    11: 'Sugar beet'
}

test_dict = {
    'image': [],
    'label': []
}

#from tqdm import tqdm
import imageio
from skimage.transform import resize as imresize

# Resize all image to 51x51 
def img_reshape(img):
    img = imresize(img, (im_size, im_size, 3))  # auto-rescales image to 0-1
    return img

# get image tag
def img_label(path):
    return str(str(path.split('/')[-1]))

# fill train and test dict
def fill_dict(path, some_dict):
    print(path)
    files = os.listdir(path)
    for p in files:  # tqdm(paths, ascii=True, ncols=85, desc='process test data'):
#        print(p)
        p_full = os.path.join(path, p)
        img = imageio.imread(p_full)
        img = img_reshape(img)
        some_dict['image'].append(img)
        some_dict['label'].append(img_label(p))

    return some_dict

#train_dict = fill_dict(train_path, train_dict)
test_dict = fill_dict(test_dir, test_dict)

X_test = np.array(test_dict['image'])
label = test_dict['label']


np.random.seed(seed)
df = None

with tf.device('/gpu:0'):
    accs = []
    losses = []
    val_accs = []
    val_losses = []
    test_accs = []
    test_losses = []    
    
    for i in range(1, EPOCHS+1):
        j = i-1
        history = model.fit_generator(train_generator,
                                      steps_per_epoch=base_train_steps * REPEATS,
                                      epochs=i,
                                      validation_data=validation_generator,
                                      validation_steps=base_val_steps,
                                      verbose=1,
                                      initial_epoch=j,
                                      workers=1)

        accs = accs + history.history['acc']
        val_accs = val_accs + history.history['val_acc']
        losses = losses + history.history['loss']
        val_losses = val_losses + history.history['val_loss']
        
        print('Best Validation Accuracy:', round(max(val_accs), 6), 'Min Loss:', round(min(val_losses), 6))
        
        if val_accs[-1] >= 0.98:
            prob = model.predict(X_test, verbose=1)
            pred = prob.argmax(axis=-1)
            sub = pd.DataFrame({"file": label, "species": [INV_CLASS[p] for p in pred]})
            sub.to_csv('submission_micro_xception_BN_%d_epochs.csv' % i, index=False, header=True)
            np.savetxt('preds_micro_xception_BN_%d_epochs.csv' % i, prob, delimiter=",")
            
            try:
                # https://machinelearningmastery.com/save-load-keras-deep-learning-models/
                model.save_weights("model_BN_epoch_%d.h5" % i)
                print("Saved model to disk")
            except:
                print('Exception saving model')

        data = {'epoch': i, 'accs': accs[-1], 'val_accs': val_accs[-1], 'losses': losses[-1], 'val_losses': val_losses[-1]}
        
        if df is None:
            df = pd.DataFrame(data=data, index=[0])
        else:
            temp = pd.DataFrame(data=data, index=[0])
            df = df.append(temp, ignore_index=True)

        df.reset_index(inplace=True, drop=True)
        df_path = os.path.join(data_path, '$df_mx_l2_70_30_%d.csv' % i)
        df.to_csv(df_path)
        print(df.tail())    				