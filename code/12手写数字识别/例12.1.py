# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 08:02:41 2021

@author: 李立宗  lilizong@gmail.com
微信公众号：计算机视觉之光（微信号cvlight）
计算机视觉40个经典案例——从入门到深度学习（python+OpenCV）（待定名称）
李立宗 著     电子工业出版社
"""
import glob
import cv2
#==============准备数据========================
#读取待识别图像
o=cv2.imread("image/test2/3.bmp",0)
# images用于存储模板
images = []
# 遍历指定目录下所有子目录及模板图像
for i in range(10):    
    images.extend(glob.glob('image/'+str(i)+'/*.*'))    
#=============计算匹配值函数=====================
def getMatchValue(template,image):
    #读取模板图像
    templateImage=cv2.imread(template)
    #模板图像色彩空间转换，BGR-->灰度
    templateImage = cv2.cvtColor(templateImage, cv2.COLOR_BGR2GRAY)
    #模板图像阈值处理， 灰度-->二值
    ret, templateImage = cv2.threshold(templateImage, 0, 255, cv2.THRESH_OTSU)
    # 获取待识别图像的尺寸
    height, width = image.shape
    # 将模板图像调整为与待识别图像尺寸一致
    templateImage = cv2.resize(templateImage, (width, height))
    #计算模板图像、待识别图像的模板匹配值
    result = cv2.matchTemplate(image, templateImage, cv2.TM_CCOEFF)
    # 将计算结果返回
    return result[0][0]
#===============计算最佳匹配值及模板序号======================
# matchValue用于存储所有匹配值
matchValue = []
# 从images中逐个提取模板，并将其与待识别图像o计算匹配值
for xi in images:
    d = getMatchValue(xi,o)
    matchValue.append(d)
# print(distance)   #测试语句：看看各个距离值
# 获取最佳匹配值
bestValue=max(matchValue)
# 获取最佳匹配值对应模板编号
i = matchValue.index(bestValue)
# print(i)         #测试语句：看看匹配的模板编号
#===============计算识别结果======================
#计算识别结果
number=int(i/10)
#===============显示识别结果======================
print("识别结果:数字",number)