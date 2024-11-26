# -*- coding: utf-8 -*-
# =============================================================================
# @author: 李立宗  lilizong@gmail.com
# 微信公众号：计算机视觉之光（微信号cvlight）
# 计算机视觉核心案例实战——从入门到深度学习（python+OpenCV）
# 李立宗 著     电子工业出版社
# =============================================================================

import cv2
import numpy as np
# 嵌入过程
def embed(O,W):
    #step 1: 最低有效位置零
    OBZ=O-O%2
    #step 2: 水印处理为01值
    #水印图像处理为0/1值(需要注意，该方式不严谨)
    WB=(W/255).astype(np.uint8)
    # 更严谨方式  
    # W[W>127]=255
    # WB=W    
    #step 3: 嵌入水印的过程
    OW=OBZ+WB
    #显示原始图像、水印图像、嵌入水印的图像
    cv2.imshow("Original",O)
    cv2.imshow("wateramrk",W)
    cv2.imshow("embed",OW)
    return OW

def extract(OW):
    # step 4：获取水印图像OW的最低有效位，获取水印信息
    EWB=OW % 2
    # step 5：将二值水印图像的数值1乘以255，得到256级灰度图像
    # 将前景色由黑色（数值1）变为白色（数值255）
    EW=EWB*255
    #显示提取结果
    cv2.imshow("extractedWatermark",EW)

# 主程序
#读取原始载体图像o
O=cv2.imread("image/lena.bmp",0)
# 读取水印图像
W=cv2.imread("image/watermark.bmp",0)
# 嵌入水印
OW=embed(O,W)
extract(OW)
# 显示控制、释放窗口
cv2.waitKey()
cv2.destroyAllWindows()

