"""
@author: 李立宗  lilizong@gmail.com
微信公众号：计算机视觉之光（微信号cvlight）
计算机视觉40例——从入门到深度学习（python+OpenCV）（待定名称）
李立宗 著     电子工业出版社
"""


import cv2
import time
def detect(image,padding):
    imagex=image.copy()   #函数内部做个副本，让每个函数运行在不同的图像上        
    hog = cv2.HOGDescriptor()   #初始化方向梯度直方图描述子
    #设置SVM为一个预先训练好的行人检测器
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())  
    #调用函数detectMultiScale，检测行人对应的边框
    time_start = time.time()     #记录开始时间
    #获取（行人对应的矩形框、对应的权重）
    (rects, weights) = hog.detectMultiScale(imagex,
                        winStride=(16,16),padding=padding)    
    time_end = time.time()    #记录结束时间
    # 绘制每一个矩形框
    for (x, y, w, h) in rects:  
        cv2.rectangle(imagex, (x, y), (x + w, y + h), (0, 0, 255), 2)
    print("Padding size:",padding,",time:",time_end-time_start)
    name=str(padding[0]) + "," + str(padding[0])
    cv2.imshow(name, imagex)     #显示原始效果
image = cv2.imread("backPadding2.jpg") 
detect(image,(0,0))
detect(image,(8,8))
cv2.waitKey(0)
cv2.destroyAllWindows()