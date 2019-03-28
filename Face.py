# # coding:utf-8
# import importlib
# import sys
# importlib.reload(sys)


# import cv2
# #读取图片
# image = cv2.imread('demo.jpg')
# #灰度转换
# gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
# # 调用训练好的数据库
# face_cascade = cv2.CascadeClassifier(r'.data/haarcascades/haarcascade_frontalface_default.xml')

# #检测人脸
# faces = face_cascade.detectMultiScale(
#    gray,
#    scaleFactor = 1.15,
#    minNeighbors = 5,
#    minSize = (5,5),
#    flags = cv2.CV_HAAR_SCALE_IMAGE
# )

# print ("发现{0}个人脸!".format(len(faces)))

# for(x,y,w,h) in faces:
#    cv2.rectangle(image,(x,y),(x+w,y+w),(0,255,0),2)


# cv2.imshow("Image Title",image)
# cv2.waitKey(0)


import numpy as np  
import cv2  
  
  
face_cascade = cv2.CascadeClassifier(r'.data/haarcascades/haarcascade_frontalface_default.xml')  
eye_cascade = cv2.CascadeClassifier(r'.data/haarcascades/haarcascade_eye_tree_eyeglasses.xml')  
  
img = cv2.imread("demo.jpg")  
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  
                      
faces = face_cascade.detectMultiScale(gray,1.1,5,cv2.CASCADE_SCALE_IMAGE,(50,50),(100,100))  
  
if len(faces)>0:  
    for faceRect in faces:  
        x,y,w,h = faceRect  
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2,8,0)  
  
        roi_gray = gray[y:y+h,x:x+w]  
        roi_color = img[y:y+h,x:x+w]  
  
        eyes = eye_cascade.detectMultiScale(roi_gray,1.1,1,cv2.CASCADE_SCALE_IMAGE,(2,2))  
        for (ex,ey,ew,eh) in eyes:  
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)  
              
cv2.imshow("img",img)  
cv2.waitKey(0)  