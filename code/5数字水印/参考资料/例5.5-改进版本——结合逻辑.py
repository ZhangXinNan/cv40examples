# -*- coding: utf-8 -*-
# =============================================================================
# @author: 李立宗  lilizong@gmail.com
# 微信公众号：计算机视觉之光（微信号cvlight）
# 计算机视觉核心案例实战——从入门到深度学习（python+OpenCV）
# 李立宗 著     电子工业出版社
# =============================================================================

import cv2

def embed():
    #读取原始载体图像
    O=cv2.imread("image/lena.bmp",0)
    #最低有效位置零
    OBZ=O-O%2
    #读取水印图像
    W=cv2.imread("image/watermark.bmp",0)
    #水印图像处理为0/1值
    W[W>1]=255
    WB=W
    #嵌入水印的过程
    OBW=OBZ+WB
    #显示嵌入了水印的载体图像
    #显示原始图像、水印图像、嵌入水印的图像
    cv2.imshow("Original",O)
    cv2.imshow("wateramrk",W)
    cv2.imshow("embed",OBW)
    return OBW

def extract(OBWD):
    EWB=OBWD % 2
    EW=EWB*255
    cv2.imshow("extractedWatermark",EW)
    return EW

def destroyall():
    cv2.waitKey()
    cv2.destroyAllWindows()
    
OBWD=embed()
extract(OBWD)
destroyall()


