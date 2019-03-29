import numpy as np  
import cv2  
  
  
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')  
  
img = cv2.imread("Array Test.png")  
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  
                      
faces = face_cascade.detectMultiScale(gray,1.1,5,cv2.CASCADE_SCALE_IMAGE,(50,50),(100,100))  
print ("发现{0}个人脸!".format(len(faces)))
if len(faces)>0:  
    for faceRect in faces:  
        x,y,w,h = faceRect  
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2,8,0)  
  
        roi_gray = gray[y:y+h,x:x+w]  
        roi_color = img[y:y+h,x:x+w]  
  
        eyes = eye_cascade.detectMultiScale(roi_gray,1.1,1,cv2.CASCADE_SCALE_IMAGE,(2,2))
        print ("发现{0}个眼睛!".format(len(eyes)))  
        for (ex,ey,ew,eh) in eyes:  
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)  
              
cv2.imshow("img",img)  
cv2.waitKey(0)  