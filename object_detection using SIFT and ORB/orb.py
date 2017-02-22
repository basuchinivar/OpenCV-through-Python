# -*- coding: utf-8 -*-
"""
Created on Sun Jan 29 03:11:26 2017

@author: Basu Chinivar
"""
import cv2
import numpy as np

def orb_detector(new_image,image_template):
    image1 = cv2.cvtColor(new_image,cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(1000,1.2)
    (kp1,d1) = orb.detectAndCompute(image1,None)
    (kp2,d2) = orb.detectAndCompute(image_template,None)
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)
    matches = bf.match(d1,d2)
    
    #matches=sorted(matches,key=lambda val : val.distance())
    
    return len((matches))
    
    
image_template1 = cv2.imread('password.png')
image_template= cv2.cvtColor(image_template1,cv2.COLOR_BGR2GRAY)

cap = cv2.VideoCapture(0)


while True:
    
    #get webcam images
    ret,frame = cap.read()
    #get height and width of the webcam frame

    height,width = frame.shape[:2]
    

    #define ROI
    top_left_x = int(width/3)
    top_left_y = int((height/2)+(height/4))
    bottom_right_x = int((width/3)*2)
    bottom_right_y = int((height/2)-(height/4))
    
    #Draw a rectangular window for roi
    cv2.rectangle(frame,(top_left_x,top_left_y),(bottom_right_x,bottom_right_y),255,3)
    
    cropped = frame[bottom_right_y:top_left_y,top_left_x:bottom_right_x]
   
    frame = cv2.flip(frame,1)
    
    #get the number of sift matches
    matches = orb_detector(cropped,image_template)
    stringa = str(matches)
    cv2.putText(frame,stringa,(50,450),cv2.FONT_HERSHEY_COMPLEX,2,(255,0,0),1)
    
    threshold = 150
    if matches>threshold:
        
        cv2.putText(frame,'obj found',(50,50),cv2.FONT_HERSHEY_COMPLEX,2,(255,0,87),2)
        
    cv2.imshow('object detection',frame)
    if cv2.waitKey(1) & 0xFF == 13:
        break
    
cap.release()

cv2.destroyAllWindows()