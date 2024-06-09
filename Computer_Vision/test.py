# test file for testing classes and functions
# TODO
# Ensure that all images are resized to a consistant size

import LPRL
from PIL import Image as im
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import random

def noisy(noise_typ,image):
   if noise_typ == "gauss":
      row,col= image.shape
      mean = 128
      var = 0.1
      sigma = var**0.5
      gauss = np.random.normal(mean,sigma,(row,col))
      gauss = gauss.reshape(row,col)
      noisy = image + gauss
      return noisy
   elif noise_typ == "s&p":
      row,col = image.shape
      s_vs_p = 0.5
      amount = 0.004
      out = np.copy(image)
      # Salt mode
      num_salt = np.ceil(amount * image.size * s_vs_p)
      coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
      out[coords] = 255

      # Pepper mode
      num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
      coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
      out[coords] = 0


      return out
   elif noise_typ == "poisson":
      vals = len(np.unique(image))
      vals = 2 ** np.ceil(np.log2(vals))
      noisy = np.random.poisson(image * vals) / float(vals)
      return noisy
   elif noise_typ =="speckle":
      row,col = image.shape
      gauss = np.random.randn(row,col)
      gauss = gauss.reshape(row,col)        
      noisy = image + image * gauss
      return noisy


image = cv.imread("test4.jpg", cv.IMREAD_GRAYSCALE)
image = cv.resize(image,(940,627))
# Works Well: 8, 11, 6, 11, 4


instance = LPRL.imMan(image)

imagenames = ['y_1.png','a_1.png','b_1.png','0_1.png','0_2.png','a_2.png']

binarised_segments = instance.binary_image
#instance.fill()
for i in range(0,len(binarised_segments)):
    plt.subplot(1,7,i+1)
    plt.imshow(binarised_segments[i],cmap='gray',interpolation='bicubic')
    plt.xticks([]), plt.yticks([])
j = 0
for i in range(0, 6):
    name = f"test{i}.png"
    #tempimage = noisy("s&p", binarised_segments[j])
    tempimage = binarised_segments[j]
    h, w = tempimage.shape
    m1 = random.randint(0,20)
    m2 = random.randint(0,20)
    M = np.float32([[1,0,m1],[0,1,m2]])
    #res = cv.warpAffine(tempimage,M,(w,h))
   #  plt.imshow(tempimage)
   #  plt.show()
    cv.imwrite(name,tempimage)
    j += 1
    if (j == len(binarised_segments) ):
        j = 0
         


plt.show()

