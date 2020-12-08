#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 09:14:42 2020

@author: sidu
"""

import os
import datetime
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.metrics import roc_auc_score

PATH = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.abspath(os.path.join(PATH, os.pardir))
CURR_TIME = (datetime.datetime.now()).strftime("%Y-%m-%d %H:%M:%S")


class CNN(object):
    def __init__(self):
        pass
    #------------------------------------
    # Preprocessing steps for data before learning 
    #-------------------
    def data_preprocessing(self):
        
        #Directories for all of the train/validation data
        fail_train_dir = os.path.join('data/train/fail')
        pass_train_dir = os.path.join('data/train/pass')
        fail_valid_dir = os.path.join('data/valid/fail')
        pass_valid_dir = os.path.join('data/valid/fail')
        
        
        
        train_datagen = ImageDataGenerator(rescale=1/255)
        validation_datagen = ImageDataGenerator(rescale=1/255)
        
        train_generator = train_datagen.flow_from_directory(
            '/data/train/',  # This is the source directory for training images
            classes = ['fail', 'pass'],
            target_size=(200, 200),  # All images will be resized to 200x200
            batch_size=120,
            # Use binary labels
            class_mode='binary')
        
        validation_generator = validation_datagen.flow_from_directory(
            '/data/valid/',  # This is the source directory for training images
            classes = ['fail', 'pass'],
            target_size=(200, 200),  # All images will be resized to 200x200
            batch_size=19,
            # Use binary labels
            class_mode='binary',
            shuffle=False)
        return train_generator, validation_generator
    
    def build_train_model(self, train_generator, validation_generator):
        model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape = (200,200,3)), 
                                        tf.keras.layers.Dense(128, activation=tf.nn.relu), 
                                        tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)])
        model.summary()
        
        model.compile(optimizer = tf.optimizers.Adam(),
                  loss = 'binary_crossentropy',
                  metrics=['accuracy'])
        
        history = model.fit(train_generator,
          steps_per_epoch=8,  
          epochs=15,
          verbose=1,
          validation_data = validation_generator,
          validation_steps=8)
        
        return model
    
    def evaluate(self, model):
        model.evaluate(validation_generator)
        
        STEP_SIZE_TEST=validation_generator.n//validation_generator.batch_size
        validation_generator.reset()
        preds = model.predict(validation_generator,
                          verbose=1)
        
        fpr, tpr, _ = roc_curve(validation_generator.classes, preds)
        roc_auc = auc(fpr, tpr)
        
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.show()
    def run(self):
        
        train_generator, validation_generator = self.data_preprocessing()
       
        
        
        evaluate(model)

def main():
    cnn = CNN()
    train_generator, validation_generator = cnn.data_preprocessing()
    model = cnn.build_train_model(train_generator, validation_generator)
    
    
    
    '''
    SAVE MODEL?
    '''
    saved_model_path = os.path.join("/models/model_"+CURR_TIME)
    tf.saved_model.save(model, saved_model_path)
    
    
    
if __name__ == "__main__":
    main()

    
    
    



