# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 23:53:04 2020

@author: Ahmed_Abohgeazy
"""



import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn import preprocessing
import pandas as pd
import random

class DataLoader():
    def prepare_data(self,img_rows,img_cols):
        self.img_rows = img_rows
        self.img_cols = img_cols        
        # Training data        
        X_tr=[]           # variable to store entire dataset        
        from tqdm import tqdm        
        ls_path = os.path.join("training_samples/Thumb Down")
        listing = os.listdir(ls_path)        
        for ls in tqdm(listing):
            listing_stop = sorted(os.listdir(os.path.join(ls_path,ls))) 
            frames = []
            img_depth=0
            for imgs in listing_stop:
                if img_depth <16:
                    img = os.path.join(os.path.join(ls_path,ls),imgs)
                    frame = cv2.imread(img)
                    frame=cv2.resize(frame,(img_rows,img_cols),interpolation=cv2.INTER_AREA)
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(gray)
                    img_depth=img_depth+1
                else:
                    break
            input_img = np.array(frames)
            ipt=np.rollaxis(np.rollaxis(input_img,2,0),2,0)
            ipt=np.rollaxis(ipt,2,0)
            X_tr.append(ipt)        
        print (ipt.shape)
        num_samples = len(X_tr) 
        print (num_samples)        
        from tqdm import tqdm        
        ls_path_2 = os.path.join("training_samples/Thumb Up")
        listing_2 = os.listdir(ls_path_2)        
        for ls_2 in tqdm(listing_2):
            listing_stop_2 = sorted(os.listdir(os.path.join(ls_path_2,ls_2)))
            frames = []
            img_depth= 0
            for imgs_2 in listing_stop_2:
                if img_depth < 16:
                    img = os.path.join(os.path.join(ls_path_2,ls_2),imgs_2)
                    frame = cv2.imread(img)
                    frame=cv2.resize(frame,(img_rows,img_cols),interpolation=cv2.INTER_AREA)
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(gray)
                    img_depth=img_depth+1
                else:
                    break
            input_img = np.array(frames)
            ipt=np.rollaxis(np.rollaxis(input_img,2,0),2,0)
            ipt=np.rollaxis(ipt,2,0)
            X_tr.append(ipt)        
        print (ipt.shape)
        num_samples = len(X_tr) 
        print (num_samples)        
        from tqdm import tqdm        
        ls_path_3 = os.path.join("training_samples/Drumming Fingers")
        listing_3 = os.listdir(ls_path_3)        
        for ls_3 in tqdm(listing_3):
            listing_stop_3 = sorted(os.listdir(os.path.join(ls_path_3,ls_3)))
            frames = []
            img_depth=0
            for imgs_3 in listing_stop_3:
                if img_depth <16:
                    img = os.path.join(os.path.join(ls_path_3,ls_3),imgs_3)
                    frame = cv2.imread(img)
                    frame=cv2.resize(frame,(img_rows,img_cols),interpolation=cv2.INTER_AREA)
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(gray)
                    img_depth=img_depth+1
                else:
                    break
            input_img = np.array(frames)
            ipt=np.rollaxis(np.rollaxis(input_img,2,0),2,0)
            ipt=np.rollaxis(ipt,2,0)
            X_tr.append(ipt)        
        print (ipt.shape)
        num_samples = len(X_tr) 
        print (num_samples)               
        from tqdm import tqdm        
        ls_path_4 = os.path.join("training_samples/Sliding Two Fingers Right")
        listing_4 = os.listdir(ls_path_4)        
        for ls_4 in tqdm(listing_4):
            listing_stop_4 = sorted(os.listdir(os.path.join(ls_path_4,ls_4)))
            frames = []
            img_depth=0
            for imgs_4 in listing_stop_4:
                if img_depth <16:
                    img = os.path.join(os.path.join(ls_path_4,ls_4),imgs_4)   
                    frame = cv2.imread(img)
                    frame=cv2.resize(frame,(img_rows,img_cols),interpolation=cv2.INTER_AREA)        
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(gray)
                    img_depth=img_depth+1
                else:
                    break
            input_img = np.array(frames)
            ipt=np.rollaxis(np.rollaxis(input_img,2,0),2,0)
            ipt=np.rollaxis(ipt,2,0)
            X_tr.append(ipt)        
        print (ipt.shape)
        num_samples = len(X_tr) 
        print (num_samples)
        from tqdm import tqdm
        
        ls_path_5 = os.path.join("training_samples/Sliding Two Fingers Left")
        listing_5 = os.listdir(ls_path_5)
        
        for ls_5 in tqdm(listing_5):
            listing_stop_5 = sorted(os.listdir(os.path.join(ls_path_5,ls_5)))
            frames = []
            img_depth=0
            for imgs_5 in listing_stop_5:
                if img_depth <16:
                    img = os.path.join(os.path.join(ls_path_5,ls_5),imgs_5)
                    frame = cv2.imread(img)
                    frame=cv2.resize(frame,(img_rows,img_cols),interpolation=cv2.INTER_AREA)
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(gray)
                    img_depth=img_depth+1
                else:
                    break
            input_img = np.array(frames)
            ipt=np.rollaxis(np.rollaxis(input_img,2,0),2,0)
            ipt=np.rollaxis(ipt,2,0)
            X_tr.append(ipt)        
        print (ipt.shape)
        num_samples = len(X_tr) 
        print (num_samples)        
        from tqdm import tqdm       
        ls_path_6 = os.path.join("training_samples/No gesture")
        listing_6 = os.listdir(ls_path_6)       
        for ls_6 in tqdm(listing_6):
            listing_stop_6 = sorted(os.listdir(os.path.join(ls_path_6,ls_6)))
            frames = []
            img_depth=0
            for imgs_6 in listing_stop_6:
                if img_depth <16:
                    img = os.path.join(os.path.join(ls_path_6,ls_6),imgs_6)
                    frame = cv2.imread(img)
                    frame=cv2.resize(frame,(img_rows,img_cols),interpolation=cv2.INTER_AREA)
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(gray)
                    img_depth=img_depth+1
                else:
                    break
            input_img = np.array(frames)
            ipt=np.rollaxis(np.rollaxis(input_img,2,0),2,0)
            ipt=np.rollaxis(ipt,2,0)
            X_tr.append(ipt)
        
    
        return X_tr
            
        
               