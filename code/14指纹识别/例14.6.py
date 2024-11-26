# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 21:04:42 2021

@author: 李立宗  lilizong@gmail.com
微信公众号：计算机视觉之光（微信号cvlight）
计算机视觉40个经典案例——从入门到深度学习（python+OpenCV）（待定名称）
李立宗 著     电子工业出版社
"""

import cv2
#===============计算两个指纹间匹配点的个数====================
def verification(src, model):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(src, None)
    kp2, des2 = sift.detectAndCompute(model, None)
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    ok = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            ok.append(m)
    num = len(ok)
    # print(num)
    if  num >= 500:   
        result= "认证通过"
    else:
        result= "认证失败"
    return result

#==============主函数====================
if __name__ == "__main__":
    src1=cv2.imread(r"verification\src1.bmp")
    src2=cv2.imread(r"verification\src2.bmp")
    model=cv2.imread(r"verification\model.bmp")
    result1=verification(src1,model)
    result2=verification(src2,model)
    print("src1验证结果为：",result1)
    print("src2验证结果为：",result2)
    cv2.imshow("src1",src1)
    cv2.imshow("src2",src2)
    cv2.imshow("model",model)
    cv2.waitKey()
    cv2.destroyAllWindows()