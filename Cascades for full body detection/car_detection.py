# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 00:31:06 2017

@author: Basu Chinivar
"""

import cv2
import numpy as np
cars_classifier = cv2.CascadeClassifier('haarcascade_car.xml')
cap = cv2.VideoCapture('cars.avi')

while cap.isOpened():
    ret , frame = cap.read()
    gray = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)
    
    cars = cars_classifier.detectMultiScale(gray,1.3,3)
    
    for (x,y,h,w) in cars:
        cv2.rectangle(frame,(x,y),(x+h,y+w),(0,255,50),3)
        cv2.imshow("cars",frame)
        
    if cv2.waitKey(1) == 13:
        break
    
cap.release()
cv2.destroyAllWindows()


import cv2
import numpy
numpy.set_printoptions(threshold=numpy.nan)

k = numpy.arange(10)
train_labels = numpy.repeat(k,250)[:,numpy.newaxis]
print(train_labels)