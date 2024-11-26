# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 07:53:41 2021

@author: Administrator
"""
import cv2
img = cv2.imread('b.jpg')
# cv2.imshow("orgin",img)
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# cv2.imshow("gray",gray)
gaussian_bulr = cv2.GaussianBlur(gray, (5, 5), 0)
# cv2.imshow("gaussian",gaussian_bulr)
edged=cv2.Canny(gaussian_bulr,50,200) 
# cv2.imshow("edged",edged)
cts, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, 
cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img, cts, -1, (0,0,255), 3)
cv2.imshow("img",img)
cv2.waitKey()
cv2.destroyAllWindows()
