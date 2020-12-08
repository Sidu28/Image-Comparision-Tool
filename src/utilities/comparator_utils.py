#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 10:32:37 2020

@author: sidu
"""

import os 
import cv2
import numpy as np
import datetime
import logging


CURR_TIME = (datetime.datetime.now()).strftime("%Y-%m-%d %H:%M:%S")




class Utils(object):
    
    def __init__(self):
        pass
    
    #------------------------------------
    # build a directory to store images, etc. 
    #-------------------
    def make_dir(self, parent_dir,dirname):  
        path = os.path.join(parent_dir, dirname) 
        os.makedirs(path, exist_ok=True) 
        return path
     
    #------------------------------------
    # Builds a log file for every run 
    #-------------------
    def Logfile_Generator(self, parent_dir, current_dir):
        target_dir = os.path.join(parent_dir,'logs')     #SEE IF THERE IS A WAY NOT TO HARD CODE
        logging.basicConfig(filename=os.path.join(target_dir,"logfilename_"+CURR_TIME+".log"), level=logging.INFO)
    
    
    #------------------------------------
    # uses absdiff to compare images and generates a absdiff result image 
    #-------------------
    def compareImages(self,img1, img2, image_name, path):
        print(image_name)
        diff = cv2.absdiff(img1, img2)
        mask = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    
        th = 1
        imask =  mask>1
    
        canvas = np.zeros_like(img2, np.uint8)
        canvas[imask] = img2[imask]
        
        dirname = "IMAGE_RESULTS_"+CURR_TIME
        path = os.path.join(path,'image_outputs')
        target_dir = self.make_dir(path, dirname)
       
        
        cv2.imwrite(os.path.join(target_dir,"result"+CURR_TIME+"_"+image_name), canvas)
        
    
    def extract_diff(self,imageA, imageB, image_name, path):
        '''
        Find the different between two image:
            + Input: two RGB image
            + Output: binary image show different between two image
        Assume the different between two image in each channel will be bigger or equal 30
        '''
        subtract = imageB.astype(np.float32) - imageA.astype(np.float32)
        mask = cv2.inRange(np.abs(subtract),(30,30,30),(255,255,255))
        
        th = 1
        imask =  mask>1
    
        canvas = np.zeros_like(imageA, np.uint8)
        canvas[imask] = imageA[imask]
        dirname = "IMAGE_RESULTS_"+CURR_TIME
        cv2.imwrite(dirname+"/result"+CURR_TIME+"_CVVVVV_"+image_name, canvas)
        
        # mask[mask_motion==255] = 1 # scale to 1 to reduce computation
        
      
        
    
    
    
    
    