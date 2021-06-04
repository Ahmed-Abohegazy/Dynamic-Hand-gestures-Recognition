import cv2
import numpy as np
from  pynput import mouse, keyboard
from pynput.keyboard import Key


#please note that the following contolls are adjusted to work with spotify key shortcuts 
keyboards= keyboard.Controller()
mouses = mouse.Controller()
def index_threshhold(conf, index,pre):
    if index ==1:
       
        print(conf[0])
        if pre == 1:
            return(index)
        else:
            return(5)
    elif index == 0:
        
        print(conf[0])
        if pre==0:
            
            return(index)
        else:
            return(5)
    elif index == 2 :
        #print(2)
        print(conf[0])
        if conf[0] >0.85:
            
            return(index)
        else:
            return(5)
    elif index==3:

        print(conf[0])
        if conf[0] >0.95:
            return(index)
        else:
            return(5)
    elif index ==4:
      
        print(conf[0])
        if conf[0] >0.96:
            return(index)
        else:
            return(5)
    elif index== 5:
            return(5)
def puttext_on(conf, index, classes,frame, font):   
    cv2.putText(frame, 'confidence: ' + str(conf), (10, 100), font, 0.7, (0, 255, 0), 2, 1)
    cv2.putText(frame, 'gesture class: ' + classes[index], (10, 50), font, 0.7, (0, 255, 0), 2, 1)

def detect_fist(frame,fistDetect ):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fist = fistDetect.detectMultiScale(gray, 1.3, 5);

def controll_PC(index):
    if index ==0:
        keyboards.press(Key.ctrl_r)
        keyboards.press(Key.down)
        keyboards.release(Key.ctrl_r)
        keyboards.release(Key.down)
        keyboards.press(Key.ctrl_r)
        keyboards.press(Key.down)
        keyboards.release(Key.ctrl_r)
        keyboards.release(Key.down)
        return('volume down')
    elif index ==1:
        keyboards.press(Key.ctrl_r)
        keyboards.press(Key.up)
        keyboards.release(Key.ctrl_r)
        keyboards.release(Key.up)
        keyboards.press(Key.ctrl_r)
        keyboards.press(Key.up)
        keyboards.release(Key.ctrl_r)
        keyboards.release(Key.up)
        return('volume up')
    elif index ==2:
        keyboards.press(Key.space)
        keyboards.release(Key.space)
        return("play/stop")
    elif index ==3:
        keyboards.press(Key.ctrl_r)
        keyboards.press(Key.right)
        keyboards.release(Key.ctrl_r)
        keyboards.release(Key.right)
      
        return('previous')
    elif index ==4:
        keyboards.press(Key.ctrl_r)
        keyboards.press(Key.left)
        keyboards.release(Key.ctrl_r)
        keyboards.release(Key.left)
        return('next')