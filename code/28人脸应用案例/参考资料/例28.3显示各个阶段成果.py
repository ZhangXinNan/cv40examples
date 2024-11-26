
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 14:38:03 2021

@author: Administrator
"""
import cv2
import dlib
import numpy as np
# ==================点集处理==================
# 关键点分配，五官的起止索引
JAW_POINTS = list(range(0, 17))
RIGHT_BROW_POINTS = list(range(17, 22))
LEFT_BROW_POINTS = list(range(22, 27))
NOSE_POINTS = list(range(27, 35))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
MOUTH_POINTS = list(range(48, 61))
FACE_POINTS = list(range(17, 68))
# 关键点集
POINTS = [LEFT_BROW_POINTS + RIGHT_EYE_POINTS +
          LEFT_EYE_POINTS +RIGHT_BROW_POINTS + NOSE_POINTS + MOUTH_POINTS]
# 处理为元组，后续使用方便
POINTStuple=tuple(POINTS)
# =================自定义函数：获取脸部（脸部掩码）=================
def getFaceMask(im, keyPoints):
    im = np.zeros(im.shape[:2], dtype=np.float64)
    for group in POINTS:
        points = cv2.convexHull(keyPoints[group])   #获取凸包
        cv2.fillConvexPoly(im, points, color=1)  #填充凸包        
    # 单通道im构成3通道im（3，行，列），改变形状（行、列、3）适应OpenCV
    # 原有形状：（3、高、宽），改变形状为（高、宽、3）
    im = np.array([im, im, im]).transpose((1, 2, 0))
    ksize=(15,15)
    im = cv2.GaussianBlur(im,ksize, 0)
    return im
# =========自定义函数：根据二人的脸部关键点集，构建映射矩阵M===========    
def getM(points1, points2):
    # 调整数据类型
    points1 = points1.astype(np.float64)
    points2 = points2.astype(np.float64)
    # 归一化：(数值-均值)/标准差
    # 计算均值
    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)
    # 减去均值
    points1 -= c1
    points2 -= c2
    # 计算标准差
    s1 = np.std(points1)
    s2 = np.std(points2)
    # 除标准差
    points1 /= s1
    points2 /= s2
    # 奇异值分解，Singular Value Decomposition
    U, S, Vt = np.linalg.svd(points1.T * points2)
    # 通过U和Vt找到R
    R = (U * Vt).T
    # 返回得到的M
    return np.vstack([np.hstack(((s2 / s1) * R,
                                       c2.T - (s2 / s1) * R * c1.T)),
                         np.matrix([0., 0., 1.])])
# =================自定义函数：获取图像关键点集=================
def getKeyPoints(im):
    rects = detector(im, 1)
    s= np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])
    return s
# =================自定义函数：统一颜色=================
def normalColor(a, b):
    ksize=(111,111)   #非常大的核，去噪等运算时为11就比较大了
    # 分别针对原始图像、目标图像进行高斯滤波 
    aGauss = cv2.GaussianBlur(a, ksize, 0)
    bGauss = cv2.GaussianBlur(b, ksize, 0)
    # 计算目标图像调整颜色的权重值
    weight= aGauss/ bGauss
    return b * weight
# =========模式初始化=============
PREDICTOR = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR)
# =============初始化：读取原始人脸a和b==============
a=cv2.imread(r"person/image2.jpg")
b=cv2.imread(r"person/image7.jpg")
bOriginal=b.copy()  #用来显示原始图像b时使用
# =========step1：获取关键点集===============
aKeyPoints = getKeyPoints(a)
bKeyPoints = getKeyPoints(b)
# =============step2:获取换脸的两个人的脸部模板===================
aMask = getFaceMask(a, aKeyPoints)
bMask = getFaceMask(b, bKeyPoints)
cv2.imshow("aMask",aMask)
cv2.imshow("bMask",bMask)
# ============step3:根据二者的关键点集构建仿射映射的M矩阵==================
M = getM(aKeyPoints[POINTStuple],bKeyPoints[POINTStuple])
# ============step4：将b的脸部（bmask）根据M仿射变换到a上==============
dsize=a.shape[:2][::-1]  
# 目标输出与图像a大小一致
# 需要注意，shape是（行、列）,warpAffine参数dsize是（列、行）
# 使用a.shape[:2][::-1]，获取a的（列、行）
bMaskWarp=cv2.warpAffine(bMask,
                   M[:2],
                   dsize,
                   borderMode=cv2.BORDER_TRANSPARENT,
                   flags=cv2.WARP_INVERSE_MAP)
cv2.imshow("bMaskWarp",bMaskWarp)
# ============step5：获取脸部最大值（两个脸模板叠加）=================
mask = np.max([aMask, bMaskWarp],axis=0)
cv2.imshow("mask",mask)
# =============step6：使用仿射矩阵M，将b映射到a===================
bWrap =cv2.warpAffine(b,
                   M[:2],
                   dsize,
                   borderMode=cv2.BORDER_TRANSPARENT,
                   flags=cv2.WARP_INVERSE_MAP) 
cv2.imshow("bWrap",bWrap)
# ==========step7:让颜色更自然一些=================
bcolor = normalColor(a, bWrap)
cv2.imshow("bcolor",bcolor/255)
# ===========step8: 换脸（mask区域用bcolor，非mask区域用a）=============
out = a * (1.0 - mask) + bcolor * mask
# =========输出原始人脸、换脸结果===============
cv2.imshow("a",a)
cv2.imshow("b",bOriginal)
cv2.imshow("out",out/255)
cv2.waitKey()
cv2.destroyAllWindows()


