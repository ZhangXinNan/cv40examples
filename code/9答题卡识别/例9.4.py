# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 19:25:22 2021

@author: 李立宗  lilizong@gmail.com
微信公众号：计算机视觉之光（微信号cvlight）
计算机视觉核心案例实战——从入门到深度学习（python+OpenCV）（名称待定）
计算机视觉40个核心案例——从入门到深度学习（python+OpenCV）（名称待定）
李立宗 著     电子工业出版社
"""

import cv2
import numpy as np
from scipy.spatial import distance as dist
# 自定义透视函数
def myWarpPerspective(image, pts):
    # step1：参数pts是要做倾斜校正的轮廓的逼近多边形（本题中的答题纸）的四个顶点，
    # 首先，确定四个顶点分别对应（左上、右上、右下、左下）的哪一个位置
    # step1.1：根据x轴值排序对4个点进行排序
    xSorted = pts[np.argsort(pts[:, 0]), :]
    #step1.2：四个点划分为：左侧2个、右侧2个
    left = xSorted[:2, :]
    right = xSorted[2:, :]
    # step1.3：在左半边寻找左上角、左下角
    # 根据y轴的值排序
    left = left[np.argsort(left[:, 1]), :]
    # 排在前面的是左上角（tl:top-left）、排在后面的是左下角（bl:bottom-left）
    (tl, bl) = left
    # step1.4：根据右侧两个点与左上角点的距离判断右侧两个点的位置
    # 计算右侧两个点距离左上角点的距离
    D = dist.cdist(tl[np.newaxis], right, "euclidean")[0]
    # 形状大致如下：
    #  左上角(tl)                 右上角(tr)
    #                页面中心
    # 左下角(bl)                   右下角(br)
    # 右侧两个点，距离左上角远的点是右下角(br)的点，近的点是右上角的点(tr)
    # br:bottom-right/tr:top-right
    (br, tr) = right[np.argsort(D)[::-1], :]
    # step1.5：确定pts的四点分别属于（左上、左下、右上、右下）的哪一个
    # src是根据（左上、左下、右上、右下）对pts的四个顶点进行排序的结果
    src = np.array([tl, tr, br, bl], dtype="float32")
    #========以下5行是测试语句，显示计算的顶点对不对=================
    # srcx = np.array([tl, tr, br, bl], dtype="int32")
    # print("看看各个顶点在哪：\n",src)   #测试语句，看看顶点
    # test=image.copy()                  #复制image，处理用
    # cv2.polylines(test,[srcx],True,(255,0,0),8)  #在test内绘制得到的点
    # cv2.imshow("image",test)                     #显示绘制线条结果    
    # =========step2：根据pts的四个顶点，计算出校正后图像的宽度和高度===============
    # 校正后图像的大小计算比较随意，根据需要选用合适值即可。
    # 这里选用较长的宽度和高度作为最终的宽度和高度
    # 计算方式：由于图像是斜的，所以通过计算x方向、y方向差值的平方根作为实际长度。
    # 具体图示如下，因为印刷原因可能对不齐，请在源代码文件中进一步看具体情况。
    #                 (tl[0],tl[1])
    #                 |\
    #                 | \    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) 
    #                 |  \                + ((tl[1] - bl[1]) ** 2))
    #                 |   \
    #                 ----- (bl[0],bl[1])
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # 根据（左上、左下）、（右上、右下）的最大值，获取高度
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # 根据宽度、高度，构造新图像dst对应的的四个顶点
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    # print("看看目标如何：\n",dst)   #测试语句
    # 构造从src到dst的仿射矩阵
    M = cv2.getPerspectiveTransform(src, dst)
    # 完成从src到dst的透视变换
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    # 返回透视变换的结果
    return warped
# 主程序
img = cv2.imread('b.jpg')
# cv2.imshow("orgin",img)
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# cv2.imshow("gray",gray)
gaussian_bulr = cv2.GaussianBlur(gray, (5, 5), 0)
# cv2.imshow("gaussian",gaussian_bulr)
edged=cv2.Canny(gaussian_bulr,50,200) 
# cv2.imshow("edged",edged)
cts, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cv2.drawContours(img, cts, -1, (0,0,255), 3)
cv2.imshow("draw_contours",img)
rightSum = 0
# 可能仅仅找到一个轮廓，就是答题纸的轮廓
# 但是，由于噪声等影响，很可能找到很多轮廓，
# 使用for循环，遍历每一个轮廓，找到答题纸的轮廓
# 将答题纸处理进行倾斜校正
for c in cts:
    peri=0.01*cv2.arcLength(c,True)
    approx=cv2.approxPolyDP(c,peri,True)
    print("顶点个数：",len(approx))
    # 四个顶点的轮廓是矩形（或者是由于扫描等原因由矩形变成的梯形）
    if len(approx)==4: 
        # 将外轮廓进行倾斜校正，将其构成一个矩形
        # 处理后，仅仅保留答题卡部分，答题卡外面的边界被删除
        # 原始图像的倾斜校正，用于后续标注
        # print(approx)
        # print(approx.reshape(4,2))
        paper = myWarpPerspective(img, approx.reshape(4, 2))
cv2.imshow("paper", paper)
cv2.waitKey()
cv2.destroyAllWindows()

