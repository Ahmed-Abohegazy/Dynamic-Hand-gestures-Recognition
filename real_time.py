
from pynput import mouse, keyboard
from pynput.keyboard import Key
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
#from keras.models import Model 
from keras.models import load_model,Model
import   controll_commands


model1_path = "saved_model/3DCNN_LRN_jester"
model1 = load_model(model1_path)
model2_path = "saved_model/3DCNN+3LSTM_jester"
model2 = load_model(model2_path)
#字型
font = cv2.FONT_HERSHEY_SIMPLEX
quietMode = False
img_rows,img_cols=100, 100 
cap = cv2.VideoCapture(0)
cv2.namedWindow('Original', cv2.WINDOW_NORMAL)

# set rt size as 640x480
ret = cap.set(3,640)
ret = cap.set(4,480)
framecount = 0
fps = ""
start = time.time()
frames = []
num=[5]
max =1
real_index = 5
instruction = 'no Gestrue'
pre =0
#load CSV
class_file = 'class_jester_6_300.csv'
with open(class_file) as f:
    classes = f.readlines()
classes = [c.strip() for c in classes]
num_classes = 6
while(1):
    ret, frame = cap.read()
    frame = cv2.flip(frame, 3)
    frame = cv2.resize(frame, (640,480))
    framecount = framecount + 1
    end  = time.time()
    timediff = (end - start)
    if( timediff >= 1):
        fps = 'FPS:%s' %(framecount)
        start = time.time()
        framecount = 0

    cv2.putText(frame,fps,(10,20), font, 0.7,(0,255,0),2,1)
    X_tr=[]
         
    image=cv2.resize(frame,(img_rows,img_cols),interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    frames.append(gray)
    input=np.array(frames)
    
    if input.shape[0]==16:
        frames = []
        X_tr.append(input)
        X_train= np.array(X_tr)  
        train_set = np.zeros((1, 16, img_cols,img_rows,3))
        train_set[0][:][:][:][:]=X_train[0,:,:,:,:]
        train_set = train_set.astype('float32')
        train_set -= 108.13708
        train_set /= 146.86292
        result_1 = model1.predict(train_set)
        result_2 = model2.predict(train_set)
        # print(result)
        num = np.argmax(result_1+result_2,axis =1)
        max = np.max((result_1+result_2)/2, axis = 1)
        print(classes[int(num[0])])
        input=[]
        real_index = controll_commands.index_threshhold(max, int(num[0]),pre)
        instruction = controll_commands.controll_PC(real_index)
        pre = int(num[0])
        controll_commands.puttext_on(max, real_index, classes, frame, font)
        cv2.putText(frame, instruction, (450, 50), font, 0.7, (0, 255, 0), 2, 1)
    if not quietMode:
            cv2.imshow('Original',frame)
    key = cv2.waitKey(1) & 0xFF
    ## Use Esc key to close the program
    if key == 27:
        break
    elif key == ord('q'):
        quietMode = not quietMode
        print("Quiet Mode - {}".format(quietMode))
cap.release()
cv2.destroyAllWindows()
    




