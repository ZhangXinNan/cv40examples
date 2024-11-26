"""
@author: 李立宗  lilizong@gmail.com
微信公众号：计算机视觉之光（微信号cvlight）
计算机视觉40例——从入门到深度学习（python+OpenCV）（待定名称）
李立宗 著     电子工业出版社
"""


import cv2
image = cv2.imread("back.jpg")            
hog = cv2.HOGDescriptor()   #初始化方向梯度直方图描述子
#设置SVM为一个预先训练好的行人检测器
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())  
#调用函数detectMultiScale，检测行人对应的边框
(rects, weights) = hog.detectMultiScale(image)
#遍历每一个矩形框，将之绘制在图像上
for (x, y, w, h) in rects:  
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
cv2.imshow("image", image)     #显示检测结果
cv2.waitKey(0)
cv2.destroyAllWindows()