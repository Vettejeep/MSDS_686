# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 18:48:12 2018
@author: Kevin Maher
"""

# this is a shell for testing models on the kaggle plant seedlings classification dataset
# needs added keras/tf model and paths updated for the environment
# https://www.kaggle.com/c/plant-seedlings-classification

import os
import math

import numpy as np
import pandas as pd

from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import backend as K

# import tensorflow items as needed for the individual model
from tensorflow.python.keras import layers
from tensorflow.python.keras.layers import Flatten
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Activation
from tensorflow.python.keras.layers import BatchNormalization
from tensorflow.python.keras.layers import Conv2D
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.layers import MaxPooling2D
from tensorflow.python.keras.layers import AveragePooling2D
from tensorflow.python.keras.layers import GlobalMaxPooling2D
from tensorflow.python.keras.applications.imagenet_utils import _obtain_input_shape
from tensorflow.python.keras import optimizers
from tensorflow.python.keras import regularizers

import tensorflow as tf

K.clear_session() 
K.set_floatx('float32')

base_dir = r'C:\Users\Vette\Desktop\Regis\#Plants2'
data_dir = r'data_micro_inception_l2_test_70_30'
data_path = os.path.join(base_dir, data_dir)
# will crash here if dir exists, that is intentional, the dir probably contains data and 
# the intent is that the 'data_path' name should be new and unique for each new run

os.mkdir(data_path)  

train_dir = os.path.join(base_dir, 'train70')
validation_dir = os.path.join(base_dir, 'validation30')
test_dir = os.path.join(base_dir, 'test_submission')

# assumes plant data split 70/30 with move_plant_data_v2.py
# sizes are based on the output of move_plant_data_v2.py script
EPOCHS = 120
LAST_EPOCH = 0
REPEATS = 10
train_size= 3319
validation_size = 1431
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



def conv2d_bn(x,
              filters,
              num_row,
              num_col,
              padding='same',
              strides=(1, 1),
              name=None,
              reg=None):
  if name is not None:
    bn_name = name + '_bn'
    conv_name = name + '_conv'
  else:
    bn_name = None
    conv_name = None
  if K.image_data_format() == 'channels_first':
    bn_axis = 1
  else:
    bn_axis = 3
  x = Conv2D(filters, (num_row, num_col),
              strides=strides,
              padding=padding,
              use_bias=False,
              kernel_regularizer=reg,
              name=conv_name)(x)
  x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
  x = Activation('relu', name=name)(x)
  return x


def build_model(input_shape=None):
    input_shape = _obtain_input_shape(input_shape,
                                        default_size=224,
                                        min_size=24,
                                        data_format=K.image_data_format(),
                                        require_flatten=False,
                                        weights='None')

    img_input = Input(shape=input_shape)
    channel_axis = 3
    reg = regularizers.l2(0.001)
    
    # layer 0
    x = conv2d_bn(img_input, 32, 3, 3, strides=(2, 2), padding='valid', reg=reg)
    x = conv2d_bn(x, 32, 3, 3, padding='valid', reg=reg)
    x = conv2d_bn(x, 64, 3, 3, reg=reg)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)
    
    x = conv2d_bn(x, 80, 1, 1, padding='valid', reg=reg)
    x = conv2d_bn(x, 192, 3, 3, padding='valid', reg=reg)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    # mixed 0, 1, 2: 35 x 35 x 256
    branch1x1 = conv2d_bn(x, 64, 1, 1)
    
    branch5x5 = conv2d_bn(x, 48, 1, 1, reg=reg)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5, reg=reg)
    
    branch3x3dbl = conv2d_bn(x, 64, 1, 1, reg=reg)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3, reg=reg)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3, reg=reg)
    
    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 32, 1, 1, reg=reg)
    x = layers.concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool],
                               axis=channel_axis, name='mixed0')


    # mixed 2: 35 x 35 x 256
    branch1x1 = conv2d_bn(x, 64, 1, 1, reg=reg)

    branch5x5 = conv2d_bn(x, 48, 1, 1, reg=reg)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5, reg=reg)

    branch3x3dbl = conv2d_bn(x, 64, 1, 1, reg=reg)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3, reg=reg)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3, reg=reg)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 64, 1, 1, reg=reg)
    x = layers.concatenate(
      [branch1x1, branch5x5, branch3x3dbl, branch_pool],
      axis=channel_axis,
      name='mixed2')

    # mixed 5, 6: 17 x 17 x 768
    for i in range(1):
        branch1x1 = conv2d_bn(x, 192, 1, 1, reg=reg)
    
        branch7x7 = conv2d_bn(x, 160, 1, 1, reg=reg)
        branch7x7 = conv2d_bn(branch7x7, 160, 1, 7, reg=reg)
        branch7x7 = conv2d_bn(branch7x7, 192, 7, 1, reg=reg)
    
        branch7x7dbl = conv2d_bn(x, 160, 1, 1, reg=reg)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1, reg=reg)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 1, 7, reg=reg)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1, reg=reg)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7, reg=reg)
    
        branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1, reg=reg)
        x = layers.concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool],
                                    axis=channel_axis,
                                    name='mixed' + str(5 + i))

    x = GlobalMaxPooling2D()(x)
    x = Flatten()(x)
    x = Dense(384, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(num_classes, activation='softmax')(x)

    model = Model(img_input, x, name='micro_xception_v1')
    opt = optimizers.Adam(lr=0.001, decay=0.001)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    return model   

model = build_model(input_shape=(im_size, im_size, 3))
# model.summary()  # check the model if desired, useful for debug

# https://www.kaggle.com/miklgr500/keras-simple-model-0-97103-best-public-score
# used the above script to define how to save data for Kaggle, did not use the model from that script
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
# end of code snippet from kaggle except for required implementation

np.random.seed(seed)
df = None

with tf.device('/gpu:0'):
    for i in range(LAST_EPOCH + 1, EPOCHS+1):
        j = i-1
        history = model.fit_generator(train_generator,
                                      steps_per_epoch=base_train_steps * REPEATS,
                                      epochs=i,
                                      validation_data=validation_generator,
                                      validation_steps=base_val_steps,
                                      verbose=1,
                                      initial_epoch=j,
                                      workers=1)

        accs = history.history['acc']
        val_accs = history.history['val_acc']
        losses = history.history['loss']
        val_losses = history.history['val_loss']
        
        if val_accs[-1] >= 0.98:
            prob = model.predict(X_test, verbose=1)
            pred = prob.argmax(axis=-1)
            sub = pd.DataFrame({"file": label, "species": [INV_CLASS[p] for p in pred]})
            
			# save kaggle submission file
            submission_path = os.path.join(data_path, 'sub_mx_l2_70_30_%d.csv' % i)
            sub.to_csv(submission_path, index=False, header=True)
            
			# save model prediction probabilities, useful if a composite model were to be made
            probability_path = os.path.join(data_path, 'preds_mx_l2_70_30_%d.csv' % i)
            np.savetxt(probability_path, prob, delimiter=",")
            
			# save model weights
            try:
                # https://machinelearningmastery.com/save-load-keras-deep-learning-models/
                weights_path = os.path.join(data_path, 'weights_mx_l2_70_30_%d.h5' % i)
                model.save_weights(weights_path)
                print("Saved model to disk")
            except:
                print('Exception saving model')
        
		# put results of run in a dataframe, allows plotting of model history later 
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
