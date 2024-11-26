# -*- coding: utf-8 -*-

"""
@author: 李立宗  lilizong@gmail.com
微信公众号：计算机视觉之光（微信号cvlight）
计算机视觉40个经典案例——从入门到深度学习（python+OpenCV）（待定名称）
李立宗 著     电子工业出版社
"""

import cv2


def location(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)     #灰度化
    ret,thresh = cv2.threshold(gray,200,255,1)      #阈值处理
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(5, 5))      #核结构
    dilated = cv2.dilate(thresh,kernel)             #膨胀
    # 获取轮廓
    mode = cv2.RETR_TREE   #轮廓检测模式
    method = cv2.CHAIN_APPROX_SIMPLE #轮廓近似方法
    contours, hierarchy = cv2.findContours(dilated,mode,method)   #提取轮廓
    #　------------   提取小单元格（9*9=81个）  ---------------
    boxHierarchy = []   #小单元格的hierarchy信息    
    imgBox=img.copy()   #用于显示每一个单元格的轮廓
    for i in range(len(hierarchy[0])):      #针对外轮廓
        if hierarchy[0][i][3] == 0:         #判断：父轮廓是外轮廓的对象（小单元格）
            boxHierarchy.append(hierarchy[0][i])   #将该符合条件轮廓的hierarchy放入boxHierarchy
            imgBox=cv2.drawContours(imgBox.copy(),contours,i,(0,0,255)) #绘制单元格轮廓            
    cv2.imshow("boxes", imgBox)          #显示每一个小单元格的轮廓，测试用
    #　-------------  提取数字边框（定位数字） -----------------
    numberBoxes=[]   #所有数字的轮廓
    imgNum=img.copy()   #用于显示数字轮廓
    for j in range(len(boxHierarchy)):
        if boxHierarchy[j][2] != -1:  #符合条件的是包含数字的小方格
            numberBox=contours[boxHierarchy[j][2]]   #小单元内数字轮廓
            numberBoxes.append(numberBox)
            x,y,w,h = cv2.boundingRect(numberBox)   #矩形包围框
            # 绘制矩形边框
            imgNum = cv2.rectangle(imgNum.copy(),(x-1,y-1),(x+w+1,y+h+1),(0,0,255),2)
    cv2.imshow("imgNum", imgNum)          #数字轮廓
    return contours , numberBoxes   #返回所有轮廓、数字轮廓
#==================主程序=====================
original = cv2.imread('x.jpg')
cv2.imshow("original",original)
contours , numberBoxes = location(original)
cv2.waitKey()
cv2.destroyAllWindows()