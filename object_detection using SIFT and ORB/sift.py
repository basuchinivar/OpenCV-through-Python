# -*- coding: utf-8 -*-
"""
Created on Sun Jan 29 01:56:42 2017

@author: Basu Chinivar
"""

import cv2
import numpy as np

def sift_matches(new_image,image_template):
    
    image1=cv2.cvtColor(new_image,cv2.COLOR_BGR2GRAY)
    image2=image_template
    
    sift = cv2.SIFT()
    
    key1 , descriptor1 = sift.detectAndCompute(image1,None)
    key2 , descriptor2 = sift.detectAndCompute(image2,None)
    
    #flann matcher
    FLANN_INDEX_KDTREE =0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE,trees=3)
    search_params = dict(checks =100)
    
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    
    matches = flann.knnMatch(descriptor1,descriptor2,k=2)
    
    good_matches=[]
    for m,n in matches:
        if m.distance<0.7 * n.distance:
            good_matches.append(m)
            
            
    return len(good_matches)





cap = cv2.VideoCapture(0)

image_template = cv2.imread('basu.jpg',0)

while True:
    
    #get webcam images
    ret , frame = cap.read()
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
    matches = sift_matches(cropped,image_template)
    cv2.putText(frame,str(matches),cv2.FONT_HERSHEY_COMPLEX,(120,120),2,255,1)
    
    threshold = 10
    if matches>threshold:
        cv2.putText(frame,'obj found',cv2.FONT_HERSHEY_COMPLEX,(10,10),2,255,2)
        
    cv2.imshow('object detection',frame)
    if cv2.waitKey(1)==13:
        break
    
    cap.release()
    cv2.destroyAllWindows()

                
    
