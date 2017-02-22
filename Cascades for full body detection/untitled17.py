# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 01:46:01 2017

@author: Basu Chinivar
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 01:32:51 2017

@author: Basu Chinivar
"""

import cv2
import numpy as np

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_classifier = cv2.CascadeClassifier('haarcascade_eye.xml')

def detector1(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    face = face_classifier.detectMultiScale(gray,1.3,3)
    print("face values are:",face)
    
    for(x,y,w,h) in face:
        x = x-50
        w = w+50
        y = y-50
        h = h+50
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,65),2)
        roi_gray = gray[y:y+h,x:x+w]
        roi_color = img[y:y+h,x:x+w]
        eyes = eye_classifier.detectMultiScale(roi_gray)
        
        for(x1,y1,w1,h1) in eyes:
            cv2.rectangle(roi_color,(x1,y1),(x1+w1,y1+h1),(0,255,65),2)
    roi_color = cv2.flip(roi_color,1)
    return roi_color
        


cap = cv2.VideoCapture(0)
while True:
    ret,frame = cap.read()
    if ret==False:
        continue
    print("there are frames",frame)

     
    cv2.imshow("face and eye",detector1(frame))
    cv2.waitKey(0)
cap.release()
cv2.destroyAllWindows()