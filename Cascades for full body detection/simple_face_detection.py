# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 00:44:22 2017

@author: Basu Chinivar
"""

import cv2

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_classifier = cv2.CascadeClassifier('haarcascade_eye.xml')
image = cv2.imread('trump.jpg')
cv2.imshow('face detection trail',image)
cv2.waitKey(0)

gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
cv2.imshow('face detection trail',gray)
cv2.waitKey(0)

face = face_classifier.detectMultiScale(gray,1.3,5)



if face is ():
    print("nan yekda")
    
for(x,y,h,w) in face:
    cv2.rectangle(image,(x,y),(x+h,y+w),(255,0,0),3)
    cv2.imshow('face',image)
    cv2.waitKey(0)
    roi_gray = gray[y:y+h,x:x+w]
    roi_color = image[y:y+h,x:x+w]
    eyes = eye_classifier.detectMultiScale(roi_gray,1.7,10)
    for(x1,y1,h1,w1) in eyes:
        cv2.rectangle(roi_color,(x1,y1),(x1+h1,y1+w1),(255,0,0),3)
        cv2.imshow('eyes',image)
        cv2.waitKey(0)
    


    
cv2.destroyAllWindows()