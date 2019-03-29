# coding:utf8
import numpy as np
import cv2
import sys
import os
import matplotlib.pyplot as plt
from PIL import Image


# 污点识别主要使用的 形态学转换， 腐蚀膨胀等找到异物
# 函数有：cv2.erode()，cv2.dilate()，cv2.morphologyEx() 等


#读取文件
# 读取文件
#imagePath = os.getcwd()+"\\demo2.png"
img = cv2.imread('wordDemo.JPG')
#转换为灰度图像
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# 利用Sobel边缘检测生成二值图
# 原理：边缘检测的原理是利用灰度变化剧烈的位置，即梯度变化剧烈的位置，Sobel算子利用差分的方法检测出边缘的位置，本质是一个卷积核乘一个矩阵

#此步骤形态学变换的预处理，得到可以查找矩形的图片
# 参数：输入矩阵、输出矩阵数据类型、设置1、0时差分方向为水平方向的核卷积，设置0、1为垂直方向,ksize：核的尺寸
sobel = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize = 3)  
# 二值化
ret, binary = cv2.threshold(sobel, 0, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY)

# 利用膨胀和腐蚀处理二值图使白色区域更方正
# 原理：腐蚀和膨胀很像卷积神经网络的池化层，不过腐蚀膨胀左右移动步长为1(为保证图大小不变)，腐蚀是取一个核中最小值、膨胀是取最大值。
# 所以对一个二值图来说，腐蚀作用是让黑色区域变多，膨胀作用是让白色区域变多。由腐蚀、膨胀作为基本操作可结合出闭运算（先膨胀后腐蚀）、开运算（先腐蚀后膨胀）、顶帽运算（图像减去开运算结果）、底帽运算（图像减去闭运算的结果）


# 设置膨胀和腐蚀操作的核函数
element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 9))
element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (24, 6))

# 膨胀一次，让轮廓突出
dilation = cv2.dilate(binary, element2, iterations = 1)

# 腐蚀一次，去掉细节，如表格线等。注意这里去掉的是竖直的线
erosion = cv2.erode(dilation, element1, iterations = 1)

# aim = cv2.morphologyEx(binary, cv2.MORPH_CLOSE,element1, 1 )   #此函数可实现闭运算和开运算
# 以上膨胀+腐蚀称为闭运算，具有填充白色区域细小黑色空洞、连接近邻物体的作用

# 再次膨胀，让轮廓明显一些
dilation2 = cv2.dilate(erosion, element2, iterations = 3)

# # 显示膨胀一次后的图像处理效果
# plt.imshow(erosion,'gray')
# plt.show()

# # 显示连续膨胀3次后的效果
# plt.imshow(dilation2,'gray')
# plt.show()


#  查找和筛选文字区域
region = []
#  查找轮廓
contours, hierarchy = cv2.findContours(dilation2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  
# 利用以上函数可以得到多个轮廓区域，存在一个列表中。
#  筛选那些面积小的
for i in range(len(contours)):
    # 遍历所有轮廓
    # cnt是一个点集
    cnt = contours[i]

    # 计算该轮廓的面积
    area = cv2.contourArea(cnt) 

    # 面积小的都筛选掉、这个1000可以按照效果自行设置
    if(area < 1000):
        continue

#     # 将轮廓形状近似到另外一种由更少点组成的轮廓形状，新轮廓的点的数目由我们设定的准确度来决定
#     # 轮廓近似，作用很小
#     # 计算轮廓长度
#     epsilon = 0.001 * cv2.arcLength(cnt, True)

#     # 
# #     approx = cv2.approxPolyDP(cnt, epsilon, True)

    # 找到最小的矩形，该矩形可能有方向
    rect = cv2.minAreaRect(cnt)
    # 打印出各个矩形四个点的位置
    print ("rect is: ")
    print (rect)

    # box是四个点的坐标
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    # 计算高和宽
    height = abs(box[0][1] - box[2][1])
    width = abs(box[0][0] - box[2][0])

    # 筛选那些太细的矩形，留下扁的
    if(height > width * 1.3):
        continue

    region.append(box)

# 用绿线画出这些找到的轮廓
for box in region:
    cv2.drawContours(img, [box], 0, (0, 255, 0), 2)


# 弹窗显示
cv2.namedWindow("img", cv2.WINDOW_NORMAL)
cv2.imshow("img", img)

# 带轮廓的图片
cv2.waitKey(0)
cv2.destroyAllWindows()