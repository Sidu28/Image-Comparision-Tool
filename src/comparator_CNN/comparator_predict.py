#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 11:19:29 2020

@author: sidu
"""


import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing import image

uploaded = #file path

for fn in uploaded.keys():
 
  # predicting images
  path = '/content/' + fn
  
  
  img = image.load_img(path, target_size=(200, 200))
  x = image.img_to_array(img)
  
  
  x = np.expand_dims(x, axis=0)
  images = np.vstack([x])
  classes = model.predict(images, batch_size=10)
  print(classes[0])
  if classes[0]<0.5:
    print(fn + " is a dandelion")
  else:
