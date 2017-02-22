# -*- coding: utf-8 -*-
"""
Created on Sun Jan 29 17:02:42 2017

@author: Basu Chinivar
"""

import cv2
import numpy as np
import imutils
from imutils.object_detection import non_max_suppression

#initiliaze the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

image = cv2.imread('pedistrians.jpg')
image = imutils.resize(image,width=min(400,image.shape[1]))
orig = image.copy()

#detect people in the image
(rects , weights) = hog.detectMultiScale(image,winStride=(4,4),padding=(0,0),scale=1.05)

#original bounding boxes
for (x,y,w,h) in rects:
    cv2.rectangle(orig,(x,y),(x+w,y+h),(0,255,0),2)
    
    #apply non maxima supression
    rects = np.array([[x,y,x+w,y+h] for (x,y,w,h) in rects])
    pick = non_max_suppression(rects,probs=None,overlapThresh=0.65)
    
    #finalBoxes
    for(xa,ya,xb,yb) in pick:
        cv2.rectangle(image,(xa,ya),(xb,yb),(0,255,0),2)
        
    
cv2.imshow("before_nms",orig)
cv2.waitKey(0)
cv2.imshow("after",image)
cv2.waitKey(0)
cv2.destroyAllWindows()
