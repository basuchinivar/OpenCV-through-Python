# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 03:52:00 2017

@author: Basu Chinivar
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 03:43:43 2017

@author: Basu Chinivar
"""

import cv2
import numpy as np
import urllib.request
import os

def store_raw():
    neg_images_link = 'http://image-net.org/api/text/imagenet.synset.geturls?wnid=n07942152'
    neg_images_urls = urllib.request.urlopen(neg_images_link).read().decode()
    
    pic_num=921
    if not os.path.exists('neg'):
        os.makedirs('neg')
    for i in neg_images_urls.split('\n'):
        try:
            print(i)
            urllib.request.urlretrieve(i,"neg/"+str(pic_num)+'.jpg')
            img = cv2.imread("neg/"+str(pic_num)+'.jpg',cv2.IMREAD_GRAYSCALE)
            resized_image = cv2.resize(img,(100,100))
            cv2.imwrite("neg/"+str(pic_num)+'.jpg',resized_image)
            pic_num+=1
        except Exception as e:
            print(str(e))
            
#store_raw()


                
"""count = 0             
for file in ['neg']:
    for img in os.listdir(file):
        print(img)
        path = 'neg/'+img
        if open("uglies/4.jpg","rb").read() == open(path,"rb").read():
            os.remove(path)"""
            
def create_pos_n_neg():        
    for file_type in ['neg']:
        for img in os.listdir(file_type):
            if file_type == 'neg':
                line = file_type+'/'+img+'\n'
                with open('bg.txt','a') as f:
                    f.write(line)
create_pos_n_neg()
            
    