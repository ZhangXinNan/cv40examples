# -*- coding: utf-8 -*-
"""
Created on Sun Aug 15 13:30:19 2021

@author: 李立宗  lilizong@gmail.com
微信公众号：计算机视觉之光（微信号cvlight）
计算机视觉40个经典案例——从入门到深度学习（python+OpenCV）（待定名称）
李立宗 著     电子工业出版社
"""

# -*- coding: utf-8 -*-
import cv2
# =================函数：在图片内显示========================
def show(img,soduko):
    height,width = img.shape[:2]        #图像高度、宽度
    box_h = height/9                    #每个数字盒体的高度
    box_w = width/9                     #每个数字盒体的宽度
    color=(0,0,255)                     #颜色
    fontFace=cv2.FONT_HERSHEY_SIMPLEX   #字体
    thickness=3                         #字体粗细    
    #---------把识别结果绘制在原始数独图像上----------------
    for i in range(9):
        for j in range(9):
            x = int(i*box_w)
            y = int(j*box_h)+40
            s = str(soduko[j][i])
            cv2.putText(img,s,(x,y),fontFace, 1, color,thickness)
    #---------显示绘制结果------------------
    cv2.imshow("soduko", img)
    cv2.imwrite("xxx.bmp",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
#==================主程序=====================
original = cv2.imread('xt.jpg')
cv2.imshow("original",original)
sudoku=[[4, 3, 6, 5, 2, 8, 7, 9, 1], [7, 2, 9, 3, 1, 4, 8, 5, 6],
        [1, 5, 8, 9, 6, 7, 4, 3, 2], [2, 8, 3, 1, 4, 5, 6, 7, 9], 
        [9, 6, 4, 7, 8, 3, 1, 2, 5], [5, 1, 7, 2, 9, 6, 3, 8, 4], 
        [6, 9, 5, 8, 7, 1, 2, 4, 3], [3, 7, 1, 4, 5, 2, 9, 6, 8], 
        [8, 4, 2, 6, 3, 9, 5, 1, 7]]
show(original,sudoku)
