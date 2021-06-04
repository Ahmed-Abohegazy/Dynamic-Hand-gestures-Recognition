# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 14:36:44 2020

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
import itertools

class ModelData():

#save trining data history  into txt file  
    def save_history(self,history,  name):
        self.history=history
        self.name=name
        print (history)
        loss=history.history['loss']
        acc=history.history['acc']
        val_loss=history.history['val_loss']
        val_acc=history.history['val_acc']
        nb_epoch=len(acc)
    
        with open(os.path.join("result", 'result_{}.txt'.format(name)), 'w') as fp:
            fp.write('epoch\tloss\tacc\tval_loss\tval_acc\n')
            for i in range(nb_epoch):
                fp.write('{}\t{}\t{}\t{}\t{}\n'.format(
                    i, loss[i], acc[i], val_loss[i], val_acc[i]))
                
# confusion plot                  
    def confusion_matrix_plot(self,cm, classes, 
                          title='Normalized Confusion Matrix', 
                          normalize=True, 
                          cmap=plt.cm.Blues):
            self.cm=cm
            self.classes=classes
            
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
            
            
    def plots(self,history):
        training_loss = history.history['loss']
        val_loss = history.history['val_loss']
        
        plt.plot(training_loss, label="training_loss")
        plt.plot(val_loss, label="validation_loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Learning Curve")
        plt.legend(loc='best')
        plt.show()
        
        training_acc = history.history['acc']
        val_acc = history.history['val_acc']
        
        plt.plot(training_acc, label="training_accuracy")
        plt.plot(val_acc, label="validation_accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("accuracy")
        plt.title("Learning Curve")
        plt.legend(loc='best')
        plt.show()