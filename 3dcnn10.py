# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 11:10:16 2020

@author: Ahmed_Abohgeazy
"""


# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 16:53:04 2020

@author: Ahmed_Abohgeazy
"""
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers.convolutional import Conv3D, MaxPooling3D,Conv2D,AveragePooling2D,AveragePooling3D 
from keras.layers import Dense, GlobalAveragePooling3D,GlobalAveragePooling2D,TimeDistributed
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler,ReduceLROnPlateau
from keras.optimizers import SGD, RMSprop, Adadelta
from keras.utils import np_utils, generic_utils
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional_recurrent import ConvLSTM2D

import theano
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn import preprocessing

# image specification
img_rows,img_cols=120, 120 

# Training data

X_tr=[]           # variable to store entire dataset

from tqdm import tqdm

ls_path = os.path.join("Thumb Down")
listing = os.listdir(ls_path)

for ls in tqdm(listing):
    listing_stop = sorted(os.listdir(os.path.join(ls_path,ls))) 
    #print(ls)
    #img_depth = len(listing_stop)
    #if img_depth < 15:
        #print(img_depth)
        #print(ls)
    frames = []
    img_depth=0
    for imgs in listing_stop:
        if img_depth <16:
            img = os.path.join(os.path.join(ls_path,ls),imgs)
            #ret, frame = cap.read()
            frame = cv2.imread(img)
            #print(frame.shape)
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frame=cv2.resize(frame,(img_rows,img_cols),interpolation=cv2.INTER_AREA)
            #print(frame.shape)
            #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(gray)
            img_depth=img_depth+1
        else:
            break
    input_img = np.array(frames)
    #print (input_img.shape)
    ipt=np.rollaxis(np.rollaxis(input_img,2,0),2,0)
    ipt=np.rollaxis(ipt,2,0)
    #print (ipt.shape)
    X_tr.append(ipt)

#X_tr_array = np.array(X_tr)   # convert the frames read into array
print (ipt.shape)
num_samples = len(X_tr) 
print (num_samples)

from tqdm import tqdm
import os
ls_path_2 = os.path.join("Thumb Up")
listing_2 = os.listdir(ls_path_2)

for ls_2 in tqdm(listing_2):
    listing_stop_2 = sorted(os.listdir(os.path.join(ls_path_2,ls_2)))
    #print(ls_2)
    #img_depth = len(listing_stop_2)
    #print(img_depth)
    frames = []
    img_depth= 0
    for imgs_2 in listing_stop_2:
        if img_depth < 16:
            img = os.path.join(os.path.join(ls_path_2,ls_2),imgs_2)
            #ret, frame = cap.read()
            #print(img)
            frame = cv2.imread(img)
            #print(imgs_2)
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frame=cv2.resize(frame,(img_rows,img_cols),interpolation=cv2.INTER_AREA)
            #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(gray)
            img_depth=img_depth+1
        else:
            break
    input_img = np.array(frames)
    #print (input_img.shape)
    ipt=np.rollaxis(np.rollaxis(input_img,2,0),2,0)
    ipt=np.rollaxis(ipt,2,0)
    #print (ipt.shape)
    X_tr.append(ipt)

#X_tr_array = np.array(X_tr)   # convert the frames read into array
print (ipt.shape)
num_samples = len(X_tr) 
print (num_samples)

from tqdm import tqdm

ls_path_3 = os.path.join("Drumming Fingers")
listing_3 = os.listdir(ls_path_3)

for ls_3 in tqdm(listing_3):
    listing_stop_3 = sorted(os.listdir(os.path.join(ls_path_3,ls_3)))
    #print(ls)
    #img_depth = len(listing_stop_3)
    frames = []
    img_depth=0
    for imgs_3 in listing_stop_3:
        if img_depth <16:
            img = os.path.join(os.path.join(ls_path_3,ls_3),imgs_3)
            #ret, frame = cap.read()
            frame = cv2.imread(img)
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frame=cv2.resize(frame,(img_rows,img_cols),interpolation=cv2.INTER_AREA)
            #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(gray)
            img_depth=img_depth+1
        else:
            break
    input_img = np.array(frames)
    #print (input_img.shape)
    ipt=np.rollaxis(np.rollaxis(input_img,2,0),2,0)
    ipt=np.rollaxis(ipt,2,0)
    #print (ipt.shape)
    X_tr.append(ipt)

#X_tr_array = np.array(X_tr)   # convert the frames read into array
print (ipt.shape)
num_samples = len(X_tr) 
print (num_samples)



from tqdm import tqdm

ls_path_4 = os.path.join("Sliding Two Fingers Right")
listing_4 = os.listdir(ls_path_4)

for ls_4 in tqdm(listing_4):
    listing_stop_4 = sorted(os.listdir(os.path.join(ls_path_4,ls_4)))

    #print(ls)
    #img_depth = len(listing_stop_3)
    frames = []
    img_depth=0
    for imgs_4 in listing_stop_4:
        if img_depth <16:
            img = os.path.join(os.path.join(ls_path_4,ls_4),imgs_4)
            #ret, frame = cap.read()
            frame = cv2.imread(img)
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frame=cv2.resize(frame,(img_rows,img_cols),interpolation=cv2.INTER_AREA)
            #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(gray)
            img_depth=img_depth+1
        else:
            break
    input_img = np.array(frames)
    #print (input_img.shape)
    ipt=np.rollaxis(np.rollaxis(input_img,2,0),2,0)
    ipt=np.rollaxis(ipt,2,0)
    #print (ipt.shape)
    X_tr.append(ipt)

#X_tr_array = np.array(X_tr)   # convert the frames read into array
print (ipt.shape)
num_samples = len(X_tr) 
print (num_samples)
from tqdm import tqdm

ls_path_5 = os.path.join("Sliding Two Fingers Left")
listing_5 = os.listdir(ls_path_5)

for ls_5 in tqdm(listing_5):
    listing_stop_5 = sorted(os.listdir(os.path.join(ls_path_5,ls_5)))

    #print(ls)
    #img_depth = len(listing_stop_3)
    frames = []
    img_depth=0
    for imgs_5 in listing_stop_5:
        if img_depth <16:
            img = os.path.join(os.path.join(ls_path_5,ls_5),imgs_5)
            #ret, frame = cap.read()
            frame = cv2.imread(img)
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frame=cv2.resize(frame,(img_rows,img_cols),interpolation=cv2.INTER_AREA)
            #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(gray)
            img_depth=img_depth+1
        else:
            break
    input_img = np.array(frames)
    #print (input_img.shape)
    ipt=np.rollaxis(np.rollaxis(input_img,2,0),2,0)
    ipt=np.rollaxis(ipt,2,0)
    #print (ipt.shape)
    X_tr.append(ipt)

#X_tr_array = np.array(X_tr)   # convert the frames read into array
print (ipt.shape)
num_samples = len(X_tr) 
print (num_samples)

from tqdm import tqdm

ls_path_6 = os.path.join("No gesture")
listing_6 = os.listdir(ls_path_6)

for ls_6 in tqdm(listing_6):
    listing_stop_6 = sorted(os.listdir(os.path.join(ls_path_6,ls_6)))
    #print(ls)
    #img_depth = len(listing_stop_3)
    frames = []
    img_depth=0
    for imgs_6 in listing_stop_6:
        if img_depth <16:
            img = os.path.join(os.path.join(ls_path_6,ls_6),imgs_6)
            #ret, frame = cap.read()
            frame = cv2.imread(img)
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frame=cv2.resize(frame,(img_rows,img_cols),interpolation=cv2.INTER_AREA)
            #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(gray)
            img_depth=img_depth+1
        else:
            break
    input_img = np.array(frames)
    #print (input_img.shape)
    ipt=np.rollaxis(np.rollaxis(input_img,2,0),2,0)
    ipt=np.rollaxis(ipt,2,0)
    #print (ipt.shape)
    X_tr.append(ipt)

#X_tr_array = np.array(X_tr)   # convert the frames read into array
print (ipt.shape)
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
  

patch_size = 16    # img_depth or number of frames used for each video

print(train_set.shape, 'train samples')


# CNN Training parameters


nb_classes = 6

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)


# Pre-processing

train_set = train_set.astype('float32')
print(np.mean(train_set))
train_set -= np.mean(train_set)
print(np.max(train_set))
train_set /=np.max(train_set)

#mobilent v2
from keras.applications import mobilenet
mobilenet_transfer=mobilenet.MobileNet(weights='imagenet', include_top=False)
model = Sequential()
model.add(TimeDistributed(mobilenet_transfer,input_shape=(patch_size,img_cols,img_rows,3)))      
model.add(Dropout(0.2))
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(nb_classes, activation='softmax'))
model.summary()


import os
#os.chdir('///home/jovyan/program/DL/11_1下tensorflow_simpson')
save_dir = os.path.join(os.getcwd(),'saved_model')
print(os.getcwd())
model_name = "3DCNN_LRN_112_6_jester"
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
checkpoint = ModelCheckpoint(model_path, monitor = 'val_acc', 
                            save_best_only=True, verbose=1)
#earlystop
earlystop = EarlyStopping(monitor='val_loss', patience=50, verbose =1)

sgd = SGD(lr=0.005,  momentum=0.9, nesterov=False)
rms = RMSprop(decay=1e-6)
ada = Adadelta(lr=0.1,decay=1e-6)
model.compile(loss='categorical_crossentropy', 
              #optimizer=sgd,
              optimizer=ada,
              metrics=['acc'])
# Split the data
X_train_new, X_val_new, y_train_new,y_val_new = train_test_split(train_set, Y_train, test_size=0.2, random_state=20)

batch_size = 32
nb_epoch =200
#steps_per_epoch=int((len(X_val_new)*1.5)/batch_size)
lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.05, 
                                cooldown=0, patience=10, min_lr=0.005/(2^4),verbose=1)
hist = model.fit(
    X_train_new,
    y_train_new,
    validation_data=(X_val_new,y_val_new),
    batch_size=batch_size,
    epochs = nb_epoch,
    shuffle=True,
    callbacks=[checkpoint,lr_reducer]
    )

training_loss = hist.history['loss']
val_loss = hist.history['val_loss']

plt.plot(training_loss, label="training_loss")
plt.plot(val_loss, label="validation_loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Learning Curve")
plt.legend(loc='best')
plt.show()

training_acc = hist.history['acc']
val_acc = hist.history['val_acc']

plt.plot(training_acc, label="training_accuracy")
plt.plot(val_acc, label="validation_accuracy")
plt.xlabel("Epochs")
plt.ylabel("accuracy")
plt.title("Learning Curve")
plt.legend(loc='best')
plt.show()

from keras.models import Model, load_model
model1_name = "3DCNN_LRN_112_6_jester"
model1_path = os.path.join(save_dir, model1_name)
model1 = load_model(model1_path)

test_pred =model1.predict(X_train_new[50:70])
result = np.argmax(test_pred, axis =1)
print(result)
img_array = X_train_new[50]

#plt.imshow(img_array[0,:,:,9])

from sklearn.metrics import confusion_matrix
met = confusion_matrix(np.argmax(y_val_new,axis =1), np.argmax(model1.predict(X_val_new),axis =1))
print(met)

import itertools
def confusion_matrix_plot(cm, classes, 
                          title='Normalized Confusion Matrix', 
                          normalize=True, 
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.around(cm, decimals=2)
        cm[np.isnan(cm)] = 0.0
    plt.subplots(1, 1, figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                  horizontalalignment="center",
                  color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
confusion_matrix_plot(met, classes=['Thumb Down', 'Thumb Up', 'Drumming Fingers', 'Sliding Two Fingers Right', 'Sliding Two Fingers Left','No Gesture'])
# from keras.models import Model, load_model
# model2_name = "3DCNN_HRN_224_5"
# model2_path = os.path.join(save_dir, model2_name)
# model2 = load_model(model2_path)
# met = confusion_matrix(np.argmax(y_val_new,axis =1), np.argmax(model1.predict(X_val_new)+model2.predict(X_val_new),axis =1))
# print(met)




