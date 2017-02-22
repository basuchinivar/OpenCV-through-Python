# -*- coding: utf-8 -*-
"""
Created on Sun Jan 29 00:49:13 2017

@author: Basu Chinivar
"""

import cv2
import numpy as np

image = cv2.imread('home.jpg')
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
orb = cv2.ORB_create(10)
keypoints = orb.detect(gray,None)

keypoints,descriptors=orb.compute(gray,keypoints)
print(keypoints)

image = cv2.drawKeypoints(image,keypoints,None,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow('final',image)
cv2.waitKey(0)
cv2.destroyAllWindows()