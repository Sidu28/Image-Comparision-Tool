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
import tesserocr
from PIL import Image
from matplotlib import pyplot as plt


CURR_TIME = (datetime.datetime.now()).strftime("%Y-%m-%d %H:%M:%S")




class Utils(object):
    
    def __init__(self):
        pass
    
    '''
    def make_dir(self, parent_dir,dirname):  
        path = os.path.join(parent_dir, dirname) 
        os.makedirs(path, exist_ok=True) 
        return path
    '''
    #------------------------------------
    # build a directory to store images, etc. 
    #-------------------
    def create_folder(self, parent_dir,dirname):  
        path = os.path.join(parent_dir, dirname) 
        os.makedirs(path, exist_ok=True) 
        return path
    
    def mkdir(self, img, image_name, path):
        dirname = "IMAGE_RESULTS_"+CURR_TIME
        path = os.path.join(path,'image_outputs')
        target_dir = self.create_folder(path, dirname)
            
        cv2.imwrite(os.path.join(target_dir,"result"+CURR_TIME+"_"+image_name), img)
        
    #------------------------------------
    # Resize Images, when the aspect ratio is the same
    #-------------------
    def imageResizeTrain(self, image, scale_factor=1024):
        height,width,cc = image.shape
        aspectRatio = width/height
        if aspectRatio < 1:
            newSize = (int(scale_factor*aspectRatio),scale_factor)
        else:
            newSize = (scale_factor,int(scale_factor/aspectRatio))
        image = cv2.resize(image,newSize)
        return image
    
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
        
        
        self.mkdir(canvas, image_name, path)        
    #------------------------------------
    # uses cv2.matchTemplate to generate an output image highlighting the best match areas 
    # between two images
    #
    #Input:
    #   img1: image matrix of N-1th run
    #   img2:  image matrix of Nth run
    #    
    #-------------------
    
    def matchTemplate(self, img1, img2, image_name, path):
        
        template = img1.copy()
        img = img2.copy()
        
        w, h = template.shape[:-1]
        
        meth = 'cv2.TM_CCOEFF'
        method = eval(meth)
        # Apply template Matching
        res = cv2.matchTemplate(img,template,method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)

        cv2.rectangle(img,top_left, bottom_right, 255, 2)
        
        
        self.mkdir(res, image_name, path)
    
    #------------------------------------
    # Returns List of all words contained in 2 image using tesseract
    #-------------------
    def tesserOCR(self, path1, path2):
            image1 = Image.open(path1)
            image2 = Image.open(path2)
            
            wordlist1 = (tesserocr.image_to_text(image1)).split() # print ocr text from image
            wordlist2 = (tesserocr.image_to_text(image2)).split()
            return(wordlist1, wordlist2)
        
    #------------------------------------
    #Uses SIFT to extract image features
    #-------------------
    def SIFT(self, imageA, imageB, image_name, path):
        
        # Helper function that performs opencv Sift extraction
        def sift_helper(img1, img2, image_name, path):
            gray1, gray2 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY),  cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
            sift = cv2.SIFT_create()
    
            kp1, des1 = sift.detectAndCompute(gray1, None)
            kp2, des2 = sift.detectAndCompute(gray2, None)
            
            #Brute Force Matching:
            
            matches = self.BF_matcher(des1, des2, kp1, kp2)
            img = cv2.drawMatchesKnn(gray1, kp1, gray2, kp2, matches, flags=2)
            '''
            
            matches, draw_params = self.FLANN_based_matcher(des1, des2)
            img = cv2.drawMatchesKnn(gray1, kp1, gray2, kp2, matches, None, **draw_params)
            '''


            for n in range(len(matches)-1):
                x = matches[n][0]
                y = matches[n][1]
                
                query_x, query_y = x.queryIdx, y.queryIdx
                train_x, train_y = x.trainIdx, y.trainIdx
                
                #print( des1[query_x][query_y],des2[train_x][train_y])
                #print(query,train)
            '''
            matches are indexed at a point.  Index that further and you get a cv2.Dmatch.  
            '''
            

            self.mkdir(img, image_name, path)

        
        sift_helper(imageA, imageB, image_name, path)
        
    
    #------------------------------------
    #FLANN based matcher for SIFT
    #-------------------
    def FLANN_based_matcher(self, des1, des2):
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)   # or pass empty dictionary
        
        flann = cv2.FlannBasedMatcher(index_params,search_params)
        
        matches = flann.knnMatch(des2,des1,k=2)
        
        # Need to draw only good matches, so create a mask
        matchesMask = [[0,0] for i in range(len(matches))]
        
        # ratio test as per Lowe's paper
        for i,(m,n) in enumerate(matches):
            if m.distance < 0.7*n.distance:
                matchesMask[i]=[1,0]
        
        draw_params = dict(matchColor = (0,255,0),
                           singlePointColor = (255,0,0),
                           matchesMask = matchesMask,
                           flags = 0)
        
        return matches, draw_params
    
    #------------------------------------
    #Brute Force matcher for SIFT
    #-------------------
    def BF_matcher(sef, des1, des2, kp1, kp2):
        # BFMatcher with default params
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1,des2, k=2)
        
        # Apply ratio test
        good = []
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                good.append([m])
                a=len(good)
                percent=(a*100)/len(kp2)
                print("{} % similarity".format(percent))
                if percent >= 75.00:
                    print('Match Found')
                if percent < 75.00:
                    print('Match not Found')
    
            return good

    #------------------------------------
    #Uses ORB to extract image features (not as good as SIFT)
    #-------------------
    def ORB(self, imageA, imageB, image_name, path):
        img = imageA
        # Initiate ORB detector
        orb = cv2.ORB_create()
        gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # find the keypoints with ORB
        kp = orb.detect(img,None)
        img=cv2.drawKeypoints(gray,kp,img)
        
        
        self.mkdir(img, image_name, path)

        
        # compute the descriptors with ORB
        kp, des = orb.compute(img, kp)
        
        print(des.shape)
        
        
        
    
    
    
    
    
    