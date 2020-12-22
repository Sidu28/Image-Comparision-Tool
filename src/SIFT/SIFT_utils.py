#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 07:14:12 2020

@author: sidu
"""
import sys
import os
import cv2 
import csv
import pickle
import datetime
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))  #allows us to import utilities from utilities directory which is a sister-directory to comparator

from utilities.comparator_utils import Utils

#current (SIFT) directory path
PATH = os.path.dirname(os.path.abspath(__file__))
#Parent (src) directory path
SRC = os.path.abspath(os.path.join(PATH, os.pardir))
CURR_TIME = (datetime.datetime.now()).strftime("%Y-%m-%d_%H:%M:%S")



class SIFT(object):
    def __init__(self, partial_img_path1, partial_img_path2):
        self.image_name_1 = partial_img_path1
        self.image_name_2 = partial_img_path2
        
        self.imagepath_List = [self.image_name_1, self.image_name_2]   #These are partial paths of the pair of images we compare
        self.imageList = [ self.imagepath_List[0].split('/')[-1]   ,  self.imagepath_List[1].split('/')[-1]  ]   #These  are just the image names without the relative path
        
        self.sift = cv2.SIFT_create()
        self.bf = cv2.BFMatcher()
    
    #------------------------------------
    # creates and returns cv2.SIFT model
    #------------------------------------
    def computeSIFT(self,image):
        return self.sift.detectAndCompute(image, None)
    
    
    #----------------------------------------------------
    # Returns the SIFT keypoints and descriptors
    #----------------------------------------------------
    def kp_and_des(self, images):
        keypoints = []
        descriptors = []
        i = 0
        for image in images:
            print("Starting for image: " + self.imagepath_List[i])
        
            keypointTemp, descriptorTemp = self.computeSIFT(image)
            keypoints.append(keypointTemp)
            descriptors.append(descriptorTemp)
            print("  Ending for image: " + self.imagepath_List[i])
            i += 1
        
        return keypoints, descriptors

    #--------------------------------------------------
    # Stores the keypoint and descriptors for later use
    #--------------------------------------------------
    def store_kp_des(self, keypoints, descriptors, dirname):
        i = 0
        for keypoint in keypoints:
            deserializedKeypoints = []
            
            if i==0:
                filepath = SRC+"/SIFT/keypoints_and_descriptors/"+dirname+"/keypoints/" + '.'.join(self.imageList[i].split('.')[:-1]) + "_prev.txt"
            else:
                filepath = SRC+"/SIFT/keypoints_and_descriptors/"+dirname+"/keypoints/" + '.'.join(self.imageList[i].split('.')[:-1]) + "_curr.txt"

            for point in keypoint:
                temp = (point.pt, point.size, point.angle, point.response, point.octave, point.class_id)
                deserializedKeypoints.append(temp)
            with open(filepath, 'wb') as fp:
                pickle.dump(deserializedKeypoints, fp)    
            i += 1
        
        j = 0
        for descriptor in descriptors:
            if j==0:
                filepath = SRC+"/SIFT/keypoints_and_descriptors/"+dirname+"/descriptors/" + '.'.join(self.imageList[j].split('.')[:-1]) + "_prev.txt"
            else:
                filepath = SRC+"/SIFT/keypoints_and_descriptors/"+dirname+"/descriptors/" + '.'.join(self.imageList[j].split('.')[:-1]) + "_curr.txt"

            with open(filepath, 'wb') as fp:
                pickle.dump(descriptor, fp)
            j += 1

    
    #------------------------------------
    # gets keypoints from data/keypoints directory
    #-------------------     
  
    def fetchKeypointFromFile(self, i, dirname):
        if i==0:
            filepath = SRC+"/SIFT/keypoints_and_descriptors/"+dirname+"/keypoints/" + '.'.join(self.imageList[i].split('.')[:-1]) + "_prev.txt"
        else:
            filepath = SRC+"/SIFT/keypoints_and_descriptors/"+dirname+"/keypoints/" + '.'.join(self.imageList[i].split('.')[:-1]) + "_curr.txt"
                      
        keypoint = []
        file = open(filepath,'rb')
        deserializedKeypoints = pickle.load(file)
        file.close()
        for point in deserializedKeypoints:
            temp = cv2.KeyPoint(x=point[0][0],y=point[0][1],_size=point[1], _angle=point[2], _response=point[3], _octave=point[4], _class_id=point[5])
            keypoint.append(temp)
        return keypoint
    
    
    #------------------------------------
    # gets descriptors from data/descriptors directory
    #------------------- 
    def fetchDescriptorFromFile(self, i, dirname):
        if i==0:
            filepath = SRC+"/SIFT/keypoints_and_descriptors/"+dirname+"/descriptors/" + '.'.join(self.imageList[i].split('.')[:-1]) + "_prev.txt"
        else:
            filepath = SRC+"/SIFT/keypoints_and_descriptors/"+dirname+"/descriptors/" + '.'.join(self.imageList[i].split('.')[:-1]) + "_curr.txt"
            
        file = open(filepath,'rb')
        descriptor = pickle.load(file)
        file.close()
        return descriptor
    
    #--------------------------------------------------------------------------------
    # File the fetches Keypoints and Descriptors and calculates the relevant scores
    #--------------------------------------------------------------------------------   
    def calculateResultsFor(self, dirname, writer, i,j):
        keypoint1 = self.fetchKeypointFromFile(i, dirname)
        descriptor1 = self.fetchDescriptorFromFile(i, dirname)
        
        keypoint2 = self.fetchKeypointFromFile(j, dirname)
        descriptor2 = self.fetchDescriptorFromFile(j, dirname)
        
        matches = self.calculateMatches(descriptor1, descriptor2)
        score = self.calculateScore(len(matches),len(keypoint1),len(keypoint2))
        
        
       # print(len(matches),len(keypoint1),len(keypoint2),len(descriptor1),len(descriptor2))
       # print(score)
        
        result_list = [self.imageList[0],score,len(matches)]
        return result_list
    
    def calculateScore(self, matches,keypoint1,keypoint2):
        return 100 * (matches/min(keypoint1,keypoint2))
    
    #------------------------------------------
    # gets matches using brute force matching
    #------------------------------------------
    def calculateMatches(self, des1,des2):
        matches = self.bf.knnMatch(des1,des2,k=2)
        topResults1 = []
        for m,n in matches:
            if m.distance < 0.7*n.distance:
                topResults1.append([m])
                
        matches = self.bf.knnMatch(des2,des1,k=2)
        topResults2 = []
        for m,n in matches:
            if m.distance < 0.7*n.distance:
                topResults2.append([m])
        
        topResults = []
        for match1 in topResults1:
            match1QueryIndex = match1[0].queryIdx
            match1TrainIndex = match1[0].trainIdx
    
            for match2 in topResults2:
                match2QueryIndex = match2[0].queryIdx
                match2TrainIndex = match2[0].trainIdx
    
                if (match1QueryIndex == match2TrainIndex) and (match1TrainIndex == match2QueryIndex):
                    topResults.append(match1)
        return topResults

    
    def run_with_resizing(self, img1, img2, dirname, writer):
        images = []
        images.append(Utils().imageResizeTrain(cv2.imread(img1)))
        images.append(Utils().imageResizeTrain(cv2.imread(img2)))
        
        keypoints, descriptors = self.kp_and_des(images)
        self.store_kp_des(keypoints, descriptors, dirname)
        return self.calculateResultsFor(dirname, writer,0,1)
    
    def run_without_resizing(self, img1, img2, dirname, writer):
        images = []
        images.append(cv2.imread(img1))
        images.append(cv2.imread(img2))
        
        keypoints, descriptors = self.kp_and_des(images)
        self.store_kp_des(keypoints, descriptors, dirname)
        return self.calculateResultsFor(dirname, writer,0,1)
        
        
    
    def run(self, img1, img2, writer):
        dirname = "kp_des_"+CURR_TIME
        Utils().create_folder(SRC+"/SIFT/keypoints_and_descriptors", dirname)
        Utils().create_folder(SRC+"/SIFT/keypoints_and_descriptors/"+dirname, 'keypoints')
        Utils().create_folder(SRC+"/SIFT/keypoints_and_descriptors/"+dirname, 'descriptors')
        list1 = self.run_with_resizing(img1, img2, dirname, writer)
        list2 = self.run_without_resizing(img1, img2, dirname, writer)
        
        writer.writerow([list1[0], list1[1], list2[1])


        
