# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 00:29:56 2020

@author: Ahmed_Abohgeazy
"""

import numpy as np
from sys import getsizeof
np.random.seed(30)
import random as rn
rn.seed(30)
from keras import backend as K
import tensorflow as tf
tf.set_random_seed(30)

from keras.models import Sequential, Model
from keras.layers import Dense,Flatten, Flatten, BatchNormalization, Activation
from keras.layers.convolutional import Conv3D, MaxPooling3D, Conv2D, MaxPooling2D
from keras.layers.recurrent import LSTM
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adadelta,Adam
from keras.layers import Dropout

from keras.applications import InceptionV3


class ModelLoader():
    def __init__(self,img_rows,img_cols,model_version,patch_size,nb_classes):

        self.img_rows = img_rows
        self.img_cols = img_cols
        self.model_version = model_version
        self.patch_size=patch_size
        self.nb_classes=nb_classes
        
        # Loads the specified model
        if self.model_version == "v1":
            print('CNN3D10 Adadelta ')            
            self.model = self.v1()
        elif self.model_version == "v2":
            print('CNN3D10 Adam')            
            self.model = self.v2()
        
        else:
            raise Exception('No model with name {} found!'.format(model_version))
        
    
    def v1(self,filtersize=(3,3,3),dense_neurons=32,dropout=0.20):
        model = Sequential()
        model.add(Conv3D(16, filtersize, padding='same',
                 input_shape=(self.patch_size,self.img_rows,self.img_cols,3)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling3D(pool_size=(2, 2, 2)))

        model.add(Conv3D(32, filtersize, padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling3D(pool_size=(2, 2, 2))) 
        model.add(Conv3D(64, filtersize, padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling3D(pool_size=(2, 2, 2)))

        model.add(Conv3D(128, filtersize, padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling3D(pool_size=(2, 2, 2)))

        model.add(Flatten())
        model.add(Dense(dense_neurons,activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout))

        model.add(Dense(dense_neurons,activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout))


        model.add(Dense(self.nb_classes,activation='softmax'))


        optimiser = Adadelta() 
        model.compile(optimizer=optimiser, loss='categorical_crossentropy', metrics=['acc'])
        return model
  
    def v2(self,filtersize=(3,3,3),dense_neurons=64,dropout=0.20):
        model = Sequential()
        model.add(Conv3D(16, filtersize, padding='same',
                  input_shape=(self.patch_size,self.img_rows,self.img_cols,3)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling3D(pool_size=(2, 2, 2)))

        model.add(Conv3D(32, filtersize, padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling3D(pool_size=(2, 2, 2)))

        model.add(Conv3D(64, filtersize, padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling3D(pool_size=(2, 2, 2)))

        model.add(Conv3D(128, filtersize, padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling3D(pool_size=(2, 2, 2)))
        
        model.add(Flatten())
        model.add(Dense(dense_neurons,activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout))

        model.add(Dense(dense_neurons,activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout))        
        model.add(Dense(self.nb_classes, activation='softmax'))
        optimizer =  Adam()
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc'])
        return model
    