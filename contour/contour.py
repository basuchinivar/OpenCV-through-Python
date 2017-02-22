# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 03:03:05 2017

@author: Basu Chinivar
"""

import cv2
import numpy as np

image = cv2.imread('messi.jpg')
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
v = np.median(gray)
#canny edges
lower = int(max(0,(1.0-0.33)*v))
upper = int(min(0,(1.0+0.33)*v))
edges = cv2.Canny(gray,lower,upper)
#cv2.imshow("canny edges",edges)
#cv2.waitKey(0)


#contour
_,contours,heirarchy =cv2.findContours(edges,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
cv2.imshow('canny edges after contouring',edges)
cv2.waitKey(0)

cv2.drawContours(image,contours,-1,(0,255,0),3)
cv2.imshow("countours",image)

cv2.waitKey(0)
cv2.destroyAllWindows()

