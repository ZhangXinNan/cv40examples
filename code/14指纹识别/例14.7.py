# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 21:04:42 2021

@author: 李立宗  lilizong@gmail.com
微信公众号：计算机视觉之光（微信号cvlight）
计算机视觉40个经典案例——从入门到深度学习（python+OpenCV）（待定名称）
李立宗 著     电子工业出版社 
"""

import os
import cv2

#===============计算两个指纹间匹配点的个数====================
def getNum(src, model):
    img1 = cv2.imread(src)  
    img2 = cv2.imread(model) 
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    ok = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            ok.append(m)
    num = len(ok)
    return num

#============获取指纹编号================
def getID(src, database):
    max = 0
    for file in os.listdir(database):
        model = os.path.join(database, file)
        num = getNum(src, model)
        print("文件名:",file,"距离：",num)
        if  max < num:
            max = num
            name = file
    ID=name[:1]
    if  max < 100:   
        ID= 9999
    return ID

#==========根据指纹编号，获取对应姓名==============
def getName(ID):
    nameID={0:'孙悟空',1:'猪八戒',2:'红孩儿',3:'刘能',4:'赵四',5:'杰克',
            6:'杰克森',7:'tonny',8:'大柱子',9:'翠花',9999:"没找到"}
    name=nameID.get(int(ID))
    return name

#==============主函数====================
if __name__ == "__main__":
    src=r"identification/src.bmp"
    database=r"identification/database"
    ID=getID(src,database)
    name=getName(ID)
    print("识别结果为：",name)