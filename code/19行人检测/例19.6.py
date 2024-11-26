"""
@author: 李立宗  lilizong@gmail.com
微信公众号：计算机视觉之光（微信号cvlight）
计算机视觉40例——从入门到深度学习（python+OpenCV）（待定名称）
李立宗 著     电子工业出版社
"""

import cv2
def detect(image,winStride,padding,scale,useMeanshiftGrouping):   
    hog = cv2.HOGDescriptor()   #初始化方向梯度直方图描述子
    #设置SVM为一个预先训练好的行人检测器
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())  
    #获取（行人对应的矩形框、对应的权重）
    (rects, weights) = hog.detectMultiScale(image,
                            winStride = winStride,
                            padding = padding,
                            scale = scale,
                            useMeanshiftGrouping=useMeanshiftGrouping)    
    # 绘制每一个矩形框
    for (x, y, w, h) in rects:  
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.imshow("result", image)     #显示原始效果
image = cv2.imread("back.jpg") 
winStride = (8,8)
padding = (2,2)
scale = 1.03
useMeanshiftGrouping=True
detect(image,winStride,padding,scale,useMeanshiftGrouping)
cv2.waitKey(0)
cv2.destroyAllWindows()