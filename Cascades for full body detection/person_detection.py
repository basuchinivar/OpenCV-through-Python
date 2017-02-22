# -*- coding: utf-8 -*-
"""
Created on Sun Jan 29 22:24:51 2017

@author: Basu Chinivar
"""

import cv2
import numpy as np
import time

body_classifier = cv2.CascadeClassifier('haarcascade_fullbody.xml')
cap = cv2.VideoCapture('walking.avi')

while cap.isOpened():
    time.sleep(1)
    _,frame = cap.read()
    
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    bodies = body_classifier.detectMultiScale(gray,1.4,3)
    
    for(x,y,w,h) in bodies:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
        cv2.imshow('Pedistrians',frame)
        
    if cv2.waitKey(1) & 0xFF == 13:
        break
    
cap.release()
cv2.destroyAllWindows()