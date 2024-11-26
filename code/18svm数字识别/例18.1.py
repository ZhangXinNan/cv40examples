# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 13:57:29 2021

@author: 李立宗  lilizong@gmail.com
微信公众号：计算机视觉之光（微信号cvlight）
计算机视觉40个经典案例——从入门到深度学习（python+OpenCV）（待定名称）
李立宗 著     电子工业出版社
"""
#=============导入库=================
import cv2
import numpy as np
#=============抗扭斜函数=================
def deskew(img):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    s=img.shape[0]
    M = np.float32([[1, skew, -0.5*s*skew], [0, 1, 0]])
    affine_flags = cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR
    size=img.shape[::-1]   
    img = cv2.warpAffine(img,M,size,flags=affine_flags)
    return img
#=============主程序=================
img=cv2.imread("rotatex.png",0)
cv2.imshow("original",img)
img=deskew(img)
cv2.imshow("result",img)
cv2.imwrite("re.bmp",img)
cv2.waitKey()
cv2.destroyAllWindows()