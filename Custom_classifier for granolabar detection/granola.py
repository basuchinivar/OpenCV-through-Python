import numpy as np
import cv2

#face_cascade = cv2.CascadeClassifier('E:\\Prof. Lander IS\\tutorials\\custom cascade\\haarcascade_frontalface_default.xml')
#eye_cascade = cv2.CascadeClassifier('E:\\Prof. Lander IS\\tutorials\\custom cascade\\haarcascade_eye.xml')

#this is the cascade we just made. Call what you want
cascade = cv2.CascadeClassifier('E:\\Prof. Lander IS\\tutorials\\custom cascade\\chinivarCascade.xml')

cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    print(ret)
    #if ret == False:
     #   break
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    watches = cascade.detectMultiScale(img,2.3, 5)
    
    # add this
    for (x,y,w,h) in watches:
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img,'Nature valley',(x-w,y-h), font, 0.5, (11,255,255), 2, cv2.LINE_AA)

    

    cv2.imshow('img',img)
    if cv2.waitKey(1) == 13:break
    

cap.release()
cv2.destroyAllWindows()