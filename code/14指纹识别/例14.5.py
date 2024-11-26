# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 10:03:14 2021

@author: 李立宗  lilizong@gmail.com
微信公众号：计算机视觉之光（微信号cvlight）
计算机视觉40个经典案例——从入门到深度学习（python+OpenCV）（待定名称）
李立宗 著     电子工业出版社
"""



import cv2
def mySift(a,b):
    sift=cv2.SIFT_create()    
    kp1, des1 = sift.detectAndCompute(a,None)
    kp2, des2 = sift.detectAndCompute(b,None)
    FLANN_INDEX_KDTREE=0
    indexParams=dict(algorithm=FLANN_INDEX_KDTREE,trees=5)
    searchParams= dict(checks=50)
    flann=cv2.FlannBasedMatcher(indexParams,searchParams)    
    matches=flann.knnMatch(des1,des2,k=2)
    good = [[m] for m, n in matches if m.distance < 0.8 * n.distance]
    resultimage = cv2.drawMatchesKnn(a, kp1, b, kp2, good, None, flags=2)
    return resultimage
if __name__ == "__main__":
    a = cv2.imread('gua1.jpg')
    b = cv2.imread('gua2.jpg')
    c = cv2.rotate(b,0)
    m1 = mySift(a, b)
    m2 = mySift(a,c)
    m3 = mySift(b,c)
    cv2.imshow("a-b",m1)
    cv2.imshow("a-c",m2)
    cv2.imshow("b-c",m3)
    cv2.waitKey()
    cv2.destroyAllWindows()
