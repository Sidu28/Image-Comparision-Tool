#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 20:46:05 2020

@author: sidu
"""
import sys
import os
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))  #allows us to import utilities from utilities directory which is a sister-directory to comparator

import json
import cv2
import datetime
import logging
import numpy as np
import tesserocr
from utilities.logging_service import LoggingService
from utilities.comparator_utils import Utils
from SIFT.SIFT_utils import SIFT


CURR_TIME = (datetime.datetime.now()).strftime("%Y-%m-%d %H:%M:%S")
PATH = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(PATH, os.pardir))

PROJECT1 = 'test1'
PROJECT2 = 'test2'




#LOG_FILENAME = "test_run_logs/logfile.log"


class Test_Comparator(object):
    
    def __init__(self, path1, path2, img1, img2):
        
        self.log = LoggingService()
        self.path1 = path1
        self.path2 = path2
        self.imgpath1 = img1
        self.imgpath2 = img2
        self.utils = Utils()
    
                
    def TestExecutionStats(self, value1,value2):
        for (k1,v1),(k2,v2) in zip(value1.items(), value2.items()): #will only loop once
        
                '''
                if(len(v1) != len(v2)):  #check if number of test execution steps is same 
                    logging.info("Mismatched number of Test Execution Stats")
                '''   
                for dict1, dict2 in zip(v1,v2):
                    
                    if(dict1['Id'] in (1,10,11,12)):   #we don't want to assert time-based values
                       continue
                    if  dict1!=dict2:
                        logging.info("Mismatched %s" % dict1["Lable"])

            

    def TestExecutionSummary(self, value1, value2):
        if value1 != value2:
            logging.info("Mismatched Test Execution Summary entries")
    
    def Steps(self,list1, list2):
        if(len(list1) != len(list2)):
            logging.info("Mismatched number of test steps. Test 1 has "+len(list1)+"steps while Test 2 has "+len(list2)+"steps.")
            raise ValueError("Unequal number of steps between runs!")
            
        for dict1, dict2 in zip(list1, list2):
            for (k1,v1),(k2,v2) in zip(dict1.items(), dict2.items()): #iterate through each dict in Steps
                if(k1=="TimeStamp"):
                    continue
                if(v1 != v2):
                    logging.info("Mismatched %s in Step Id %s" % (k1, dict1["Id"]))
                
            
        
    def TestCases(self, value1, value2):
        dict1, dict2 = value1[0], value2[0]
        for (k1,v1),(k2,v2) in zip(dict1.items(), dict2.items()):
            if k1=="Steps":
                self.Steps(v1,v2)
            
            elif k1 in ("StartTime", "EndTime", "Duration"):
                continue
            
            elif v1 != v2:
                logging.info("Mismatched TestCase %s" % k1)
    
        
 
    def run(self):
        
        #Generates Log File for 1 comparison run
        self.utils.Logfile_Generator(SRC, PATH)   
        
        #These two are the  key value of the overall JSON, like "AODocs_TestRun_05_11_2020_17_00_08"
        key1 = list((json.load( open(self.path1))).keys())[0] 
        key2 = list((json.load( open(self.path2))).keys())[0]

        #dictionaries to be queried and validated
        test1 = (json.load( open(self.path1)))[key1]
        test2 = (json.load( open(self.path2)))[key2]
        
        
        for image1, image2 in zip(sorted(os.listdir(self.imgpath1)), sorted(os.listdir(self.imgpath2))):
            
            #print(self.imgpath1+image1, self.imgpath2+image2)
            
            
            if image1 == '.DS_Store' or image2 == '.DS_Store':
                continue
            
            path1 = self.imgpath1+image1
            path2 = self.imgpath2+image2
            img1 = cv2.imread(path1)
            img2 = cv2.imread(path2)
            
            #These partial paths are the path from within the test_data directory onwards (see os.relpath())
            #We use these to help identify which images are being used in the SIFT.  
            partial_path1 = os.path.relpath(path1, SRC+'/test_data')
            partial_path2 = os.path.relpath(path2, SRC+'/test_data')
            
            
            sift = SIFT(partial_path1, partial_path2)
            sift.run(path1, path2)
            
            
            
        
            #self.utils.ORB(img1, img2, image1, SRC)
            #self.utils.SIFT(img1, img2, image1, SRC)
            #self.utils.matchTemplate(img1, img2, image1, SRC)
            #self.utils.tesserOCR(path1, path2)
            #self.utils.compareImages(img1, img2, image1, SRC)
            #self.utils.extract_diff(img1,img2, image1, PATH)
            
        
        
        for (k1,v1),(k2,v2) in zip(test1.items(), test2.items()):
            
            if(k1 == "TestExecutionStats"):
               self.TestExecutionStats(v1,v2)
            
            if(k1 == "TestExecutionSummary"):
                self.TestExecutionSummary(v1,v2)
                
                
            if(k1 == "TestCases"):
                self.TestCases(v1,v2)
                
            if(k1 in ("AccessRequestId", "RITMId", "TaskId")):
                if(v1!=v2):
                    #raise Exception("Mismatched %s" % k1)
                    logging.info("Mismatched %s" % k1)
        
    
       


    

    
def main():    
    
    path1 = os.path.join(SRC, 'test_data/test1/Output.json')
    path2 = os.path.join(SRC,'test_data/test2/Output.json')  
      
    img1 = os.path.join(SRC, 'test_data/'+PROJECT1+'/Screenshots/')
    img2 = os.path.join(SRC, 'test_data/'+PROJECT2+'/Screenshots/')
    
    
   #Create instance of Test_Comparator class and run it
    testcomp = Test_Comparator(path1, path2, img1, img2)
    testcomp.run()




if __name__ == "__main__":
    main()


    
