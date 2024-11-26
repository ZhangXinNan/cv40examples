# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 17:20:56 2021

@author: 李立宗  lilizong@gmail.com
微信公众号：计算机视觉之光（微信号cvlight）
计算机视觉40个经典案例——从入门到深度学习（python+OpenCV）（待定名称）
李立宗 著     电子工业出版社
"""
#=============导入库=================
import cv2
import numpy as np
#=============HOG函数=================
def hog(img):
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)
    bins = np.int32(16*ang/(2*np.pi))    
    bin_cells = bins[:10,:10], bins[10:,:10], bins[:10,10:], bins[10:,10:]
    mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
    hists = [np.bincount(b.ravel(), m.ravel(),16) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)    
    return hist
#=============主程序=================
img=cv2.imread("number2.bmp",0)
cv2.imshow("original",img)
img=hog(img)
print(img)
print(img.shape)
cv2.waitKey()
cv2.destroyAllWindows()