# -*- coding: utf-8 -*-
"""
Created on Sat Jan 28 22:59:46 2017

@author: Basu Chinivar
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Jan 28 22:29:34 2017

@author: Basu Chinivar
"""

import cv2
import numpy as np

image = cv2.imread('home.jpg')
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
gray - np.float32(gray)
corners =cv2.goodFeaturesToTrack(gray,1000,0.01,10)
corners = np.int0(corners)


for corner in corners:
    x,y = corner.ravel()
    cv2.circle(image,(x,y),3,255,-1)
    
cv2.imshow("corners",image)
cv2.waitKey(0)
cv2.destroyAllWindows()

