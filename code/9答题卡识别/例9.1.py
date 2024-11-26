# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 07:14:39 2021

@author: 李立宗  lilizong@gmail.com
微信公众号：计算机视觉之光（微信号cvlight）
计算机视觉核心案例实战——从入门到深度学习（python+OpenCV）（名称待定）
计算机视觉40个核心案例——从入门到深度学习（python+OpenCV）（名称待定）
李立宗 著     电子工业出版社
"""

# ==================导入库=======================
import numpy as np
import cv2
# ==================答案及选项初始化=======================
# 将选项放入字典内
ANSWER_KEY = {0: "A", 1: "B", 2: "C", 3: "D"}
# 标准答案
ANSWER = "C"
# ==================读取原始图像=======================
img = cv2.imread('xiaogang.jpg')
cv2.imshow("original",img)
# ==================图像预处理=======================
# 转换为灰度图像
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# 高斯滤波
gaussian_bulr = cv2.GaussianBlur(gray, (5, 5), 0)  
# 阈值变换，将所有选项处理为前景（白色）
ret,thresh = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
# cv2.imshow("thresh",thresh)
# cv2.imwrite("thresh.jpg",thresh)
# ==================获取轮廓及排序=======================
# 获取轮廓
cnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# 将轮廓按照从左到右排列，方便后续处理
boundingBoxes = [cv2.boundingRect(c) for c in cnts]
(cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                    key=lambda b: b[1][0], reverse=False))
# ==================构建列表，用来存储每个选项非零值个数及序号===================
options=[]
# 自左向右，遍历每一个选项的轮廓
for (j, c) in enumerate(cnts):
    # 构造一个与原始图像大小一致的灰度图像，用来保存每一个选项用
    mask = np.zeros(gray.shape, dtype="uint8")
    # 获取单个选项
    # 这里通过循环，将每一个选项单独放入一个mask中
    cv2.drawContours(mask, [c], -1, 255, -1)
    # 获取thresh中mask指定部分，每次循环，mask对应ABCD的不同选项
    cv2.imshow("s1",mask)
    cv2.imwrite("s1.jpg",mask)
    cv2.imshow("thresh",thresh)
    cv2.imwrite("thresh.jpg",thresh)
    mask = cv2.bitwise_and(thresh, thresh, mask=mask)  
    cv2.imshow("s2",mask)
    cv2.imwrite("s2.jpg",mask)
    # cv2.imshow("mask"+str(j),mask)
    # cv2.imwrite("mask"+str(j)+".jpg",mask)
    # 计算每一个选项的非零值（白色像素点）
    # 涂为答案的选项，非零值较多；没有涂选的选项，非零值较少   
    total = cv2.countNonZero(mask)
    #将选项非零值个数、选项序号放入列表options内
    options.append((total,j))
    # print(options)  #在循环中打印存储的非零值（白色点个数）及序号
# =================识别考生的选项========================
# 将所有选项按照非零值个数降序排序
options=sorted(options,key=lambda x: x[0],reverse=True)
# 获取包含最多白色像素点的选项索引（序号）
choice_num=options[0][1]
# 根据索引确定选项值：ABCD
choice= ANSWER_KEY.get(choice_num)
print("该生的选项：",choice)
# =================根据选项正确与否，用不同颜色标注考生选项==============
# 设定标注的颜色类型，绿对红错
if choice == ANSWER:
    color = (0, 255, 0)   #回答正确，用绿色表示
    msg="回答正确"
else:
    color = (0, 0, 255)   #回答错误，用红色表示
    msg="回答错误"
#  在选项位置上标注颜色(绿对红错)
cv2.drawContours(img, cnts[choice_num], -1, color, 2)
cv2.imshow("result",img)
# 打印识别结果
print(msg)
cv2.waitKey(0)
cv2.destroyAllWindows()