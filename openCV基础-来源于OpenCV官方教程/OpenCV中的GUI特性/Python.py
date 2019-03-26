#  import numpy as np 
import cv2
from matplotlib import pyplot as plt


# Load an color image in grayscale
img = cv2.imread('D:\demo.jpg',0)

plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([]) # to hide tick values on X and Y axis
plt.show()

#print(img)

