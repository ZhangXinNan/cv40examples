# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 20:55:14 2021

@author: 李立宗  lilizong@gmail.com
微信公众号：计算机视觉之光（微信号cvlight）
计算机视觉40例——从入门到深度学习（python+OpenCV）（待定名称）
李立宗 著     电子工业出版社
"""
import cv2
# ================1 加载分类器========================
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile=cv2.CascadeClassifier("haarcascade_smile.xml")
# ================2 处理摄像头视频========================
# 初始化摄像头
cap=cv2.VideoCapture(0,cv2.CAP_DSHOW)
# 处理每一帧
while True:
    # 读取一帧
    ret,image=cap.read()
    image=cv2.flip(image,1)
    # 没有读到，直接退出
    if ret is None:
        break
    # 灰度化（彩色BGR-->灰度Gray)    
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    # 人脸检测
    faces = faceCascade.detectMultiScale(gray,
                                         scaleFactor = 1.1,
                                         minNeighbors = 5,
                                         minSize = (5,5))
    # ==================处理每个人脸=======================
    for(x,y,w,h) in faces:
      cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2) 
      # 提取人脸所在区域，多通道形式
      # roiColorFace=image[y:y+h,x:x+w]
      # 提取人脸所在区域，单通道形式
      roi_gray_face=gray[y:y+h,x:x+w]
      # 微笑检测，仅在人脸区域内检测
      smiles=smile.detectMultiScale(roi_gray_face,
                                     scaleFactor = 1.5,
                                     minNeighbors = 25,
                                     minSize = (50,50))
      for (sx,sy,sw,sh) in smiles:
          # 如果显示smiles，会有很多反馈
          # cv2.rectangle(roiColorFace,(sx,sy),(sx+sw,sy+sh),
          #               color=(0,255,255),
          #               thickness=2)
          # 显示文字“smile”表示微笑了
          cv2.putText(image,"smile",(x,y),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,
                      (0,255,255),thickness=2)
    # 显示结果
    cv2.imshow("dect",image)
    key=cv2.waitKey(25)
    if key==27:
        break
cap.release()
cv2.destroyAllWindows()