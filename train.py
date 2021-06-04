# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 13:22:50 2020

@author: Ahmed_Abohgeazy
"""


import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers.convolutional import Conv3D, MaxPooling3D,Conv2D,AveragePooling2D,AveragePooling3D
from keras.layers import Dense, GlobalAveragePooling3D,GlobalAveragePooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler,ReduceLROnPlateau
from keras.optimizers import SGD, RMSprop, Adadelta
from keras.utils import np_utils, generic_utils
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional_recurrent import ConvLSTM2D

import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn import preprocessing
from data_processing import DataLoader
from plots_traindata import ModelData

# uncomment the wanted model to train 
#from CNN3D10 import ModelLoader
#from CNN3D_LRN import ModelLoader
#from CNN3D_3lstm import ModelLoader



# image specification
img_rows,img_cols=100, 100 

data=DataLoader()
X_tr=data.prepare_data(img_rows,img_cols)
num_samples = len(X_tr) 
print (num_samples)
X_tr_array = np.array(X_tr)   # convert the frames read into array
num_samples = len(X_tr_array) 
print (num_samples)
label=np.ones((num_samples,),dtype = int)
label[0:300]= 0
label[300:600] = 1
label[600:900] = 2
label[900:1199] = 3
label[1199:1499] = 4
label[1499:1799] = 5
img_depth = 16
train_data = [X_tr_array,label]
(X_train, y_train) = (train_data[0],train_data[1])
print('X_Train shape:', X_train.shape)
train_set = np.zeros((num_samples, img_depth, img_cols,img_rows,3))
for h in range(num_samples):
    train_set[h][:][:][:][:]=X_train[h,:,:,:]

print(train_set.shape, 'train samples')


nb_classes = 6
# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
# Pre-processing
train_set = train_set.astype('float32')
print(np.mean(train_set))
train_set -= np.mean(train_set)
print(np.max(train_set))
train_set /=np.max(train_set)

# Split the data
X_train_new, X_val_new, y_train_new,y_val_new = train_test_split(train_set, Y_train, test_size=0.2, random_state=20)


#model hyperparamenters 
patch_size = 16    # img_depth or number of frames used for each video
batch_size = 8
nb_epoch =50
model_version="+lstm"
weight_decay = 0.00005
lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.05, 
                                cooldown=0, patience=10, min_lr=0.005/(2^4),verbose=1)

#call the model  note when using CNN3D10 archeticture romeove weight_decay parameter from function call
# model_intialize=ModelLoader(img_rows,img_cols,model_version,patch_size,nb_classes,weight_decay)
# model=model_intialize.model
#########################################

# used to use resnet +lstm architecture
from resnet3d_LSTM import Resnet3DBuilder
model = Resnet3DBuilder.build_resnet_50((16, 100, 100, 3), 6)
model.compile(optimizer='Adam',
              loss='categorical_crossentropy',
              metrics=['acc'])
#########################################


#define the saved model wieght path / define callback [early stop checkpoint]
import os
save_dir = os.path.join(os.getcwd(),'saved_model')
print(os.getcwd())
model_name = "build_resnet_50"+model_version
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
checkpoint = ModelCheckpoint(model_path, monitor = 'val_acc', 
                            save_best_only=True, verbose=1)
#earlystop
earlystop = EarlyStopping(monitor='val_loss', patience=25, verbose =1)

hist = model.fit(
    X_train_new,
    y_train_new,
    validation_data=(X_val_new,y_val_new),
    batch_size=batch_size,
    epochs = nb_epoch,
    shuffle=True,
    callbacks=[checkpoint,lr_reducer]
    )

from keras.models import Model, load_model
model1_name = "build_resnet_50"+model_version
model1_path = os.path.join(save_dir, model1_name)
model1 = load_model(model1_path)

test_pred =model1.predict(X_train_new[50:70])
result = np.argmax(test_pred, axis =1)
print(result)
img_array = X_train_new[50]


#generate plots 
ModelData=ModelData()
save_data=ModelData.save_history(hist,model1_name)
plots=ModelData.plots(hist)

from sklearn.metrics import confusion_matrix
met = confusion_matrix(np.argmax(y_val_new,axis =1), np.argmax(model1.predict(X_val_new),axis =1))
print(met)
confusion_matrix_plot=ModelData.confusion_matrix_plot(met, classes=['Thumb Down', 'Thumb Up', 'Drumming Fingers', 'Sliding Two Fingers Right', 'Sliding Two Fingers Left','No Gesture'])
