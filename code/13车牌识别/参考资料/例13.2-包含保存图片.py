# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 07:26:46 2021

@author: 李立宗  lilizong@gmail.com
微信公众号：计算机视觉之光（微信号cvlight）
计算机视觉40个经典案例——从入门到深度学习（python+OpenCV）（待定名称）
李立宗 著     电子工业出版社
"""

import cv2
#=====读取车牌=====
image=cv2.imread("gg.bmp")
o=image.copy()  #复制原始图像，用于绘制轮廓用
cv2.imshow("original",image)
cv2.imwrite("okimage/o1.bmp",image)
#=============图像预处理===============
# -------图像去噪处理-------
image = cv2.GaussianBlur(image, (3, 3), 0)
cv2.imshow("GaussianBlur",image)
cv2.imwrite("okimage/o2.bmp",image)
# -------色彩空间转换-------
gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
cv2.imshow("gray",gray_image)
cv2.imwrite("okimage/o3.bmp",gray_image)
# -------阈值处理（二值化） -------  
ret, image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_OTSU)
cv2.imshow("threshold",image)
cv2.imwrite("okimage/o4.bmp",image)
#-------膨胀处理，让一个字构成一个整体（大多数字不是一体的，是分散的）--------
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
image = cv2.dilate(image, kernel)
cv2.imshow("dilate",image)
cv2.imwrite("okimage/o5.bmp",image)
#=============拆分车牌，将车牌内各个字符分离===============
# -------查找轮廓，各个字符的轮廓及噪声点轮廓---------------
contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
x = cv2.drawContours(o.copy(), contours, -1, (0, 0, 255), 1)
cv2.imshow("contours",x)
cv2.imwrite("okimage/o6.bmp",x) 
print(len(contours))  #测试语句：看看找到多少个轮廓
# -------------遍历所有轮廓,寻找最小包围框------------------
chars = []
for item in contours:
    rect = cv2.boundingRect(item)
    x,y,w,h = cv2.boundingRect(item)
    chars.append(rect)
    cv2.rectangle(o,(x,y),(x+w,y+h),(0,0,255),1)
cv2.imshow("contours2",o) 
cv2.imwrite("okimage/o7.bmp",o)
# --------------将包围框按照x轴坐标值排序（自左向右排序）--------------
chars = sorted(chars,key=lambda s:s[0],reverse=False)    
# --------将字符的轮廓筛选出来-------------------
#逐个遍历包围框，高宽比在1.5-8之间，宽度大于3个像素，判定为字符
plateChars = []
for word in chars:
    if (word[3] > (word[2] * 1.5)) and (word[3] < (word[2] * 8)) and (word[2] > 3):
        plateChar = image[word[1]:word[1] + word[3], word[0]:word[0] + word[2]]
        plateChars.append(plateChar)
# --------------测试语句：查看各个字符-------------------
for i,im in enumerate(plateChars):
    cv2.imshow("char"+str(i),im)
    cv2.imwrite("okimage/o8"+str(i)+".bmp",im)
cv2.waitKey()
cv2.destroyAllWindows()