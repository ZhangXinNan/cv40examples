# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 22:01:00 2021

@author: 李立宗  lilizong@gmail.com
微信公众号：计算机视觉之光（微信号cvlight）
计算机视觉核心案例实战——从入门到深度学习（python+OpenCV）（名称待定）
计算机视觉40个核心案例——从入门到深度学习（python+OpenCV）（名称待定）
李立宗 著     电子工业出版社
"""


import cv2
import numpy as np
from scipy.spatial import distance as dist
# 自定义函数，实现透视变换（倾斜校正）
def myWarpPerspective(image, pts):
    # step1：参数pts是要做倾斜校正的轮廓的逼近多边形（本题中的答题纸）的四个顶点，
    # 首先，确定四个顶点分别对应（左上、右上、右下、左下）的哪一个位置
    # step1.1：根据x轴值排序对4个点进行排序
    xSorted = pts[np.argsort(pts[:, 0]), :]
    #step1.2：四个点划分为：左侧2个、右侧2个
    left = xSorted[:2, :]
    right = xSorted[2:, :]
    # step1.3：在左半边寻找左上角、左下角
    # 根据y轴的值排序
    left = left[np.argsort(left[:, 1]), :]
    # 排在前面的是左上角（tl:top-left）、排在后面的是左下角（bl:bottom-left）
    (tl, bl) = left
    # step1.4：根据右侧两个点与左上角点的距离判断右侧两个点的位置
    # 计算右侧两个点距离左上角点的距离
    D = dist.cdist(tl[np.newaxis], right, "euclidean")[0]
    # 形状大致如下：
    #  左上角(tl)                 右上角(tr)
    #                页面中心
    # 左下角(bl)                   右下角(br)
    # 右侧两个点，距离左上角远的点是右下角(br)的点，近的点是右上角的点(tr)
    # br:bottom-right/tr:top-right
    (br, tr) = right[np.argsort(D)[::-1], :]
    # step1.5：确定pts的四点分别属于（左上、左下、右上、右下）的哪一个
    # src是根据（左上、左下、右上、右下）对pts的四个顶点进行排序的结果
    src = np.array([tl, tr, br, bl], dtype="float32")
    #========以下5行是测试语句，显示计算的顶点对不对=================
    # srcx = np.array([tl, tr, br, bl], dtype="int32")
    # print("看看各个顶点在哪：\n",src)   #测试语句，看看顶点
    # test=image.copy()                  #复制image，处理用
    # cv2.polylines(test,[srcx],True,(255,0,0),8)  #在test内绘制得到的点
    # cv2.imshow("image",test)                     #显示绘制线条结果    
    # =========step2：根据pts的四个顶点，计算出校正后图像的宽度和高度===============
    # 校正后图像的大小计算比较随意，根据需要选用合适值即可。
    # 这里选用较长的宽度和高度作为最终的宽度和高度
    # 计算方式：由于图像是斜的，所以通过计算x方向、y方向差值的平方根作为实际长度。
    # 具体图示如下，因为印刷原因可能对不齐，请在源代码文件中进一步看具体情况。
    #                 (tl[0],tl[1])
    #                 |\
    #                 | \    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) 
    #                 |  \                + ((tl[1] - bl[1]) ** 2))
    #                 |   \
    #                 ----- (bl[0],bl[1])
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # 根据（左上、左下）、（右上、右下）的最大值，获取高度
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # 根据宽度、高度，构造新图像dst对应的的四个顶点
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    # print("看看目标如何：\n",dst)   #测试语句
    # 构造从src到dst的仿射矩阵
    M = cv2.getPerspectiveTransform(src, dst)
    # 完成从src到dst的透视变换
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    # 返回透视变换的结果
    return warped
# 标准答案
ANSWER = {0: 1, 1: 2, 2: 0, 3: 2, 4: 3}
# 答案用到的字典
answerDICT = {0: "A", 1: "B", 2: "C", 3: "D"}
# 读取原始图像（考卷）
img = cv2.imread('b.jpg')
# cv2.imshow("orgin",img)
# 图像预处理：色彩空间变换
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# cv2.imshow("gray",gray)
# 图像预处理：高斯滤波
gaussian_bulr = cv2.GaussianBlur(gray, (5, 5), 0)
# cv2.imshow("gaussian",gaussian_bulr)
# 图像预处理：边缘检测
edged=cv2.Canny(gaussian_bulr,50,200) 
# cv2.imshow("edged",edged)
# 查找轮廓
cts, hierarchy = cv2.findContours( edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cv2.drawContours(img, cts, -1, (0,0,255), 3)
# 轮廓排序
list=sorted(cts,key=cv2.contourArea,reverse=True)
print("寻找轮廓的个数：",len(cts))
# cv2.imshow("draw_contours",img)
rightSum = 0
# 可能仅仅找到一个轮廓，就是答题纸的轮廓
# 但是，由于噪声等影响，很可能找到很多轮廓，
# 使用for循环，遍历每一个轮廓，找到答题纸的轮廓
# 将答题纸处理进行倾斜校正
for c in list:
    peri=0.01*cv2.arcLength(c,True)
    approx=cv2.approxPolyDP(c,peri,True)
    print("顶点个数：",len(approx))
    # 四个顶点的轮廓是矩形（或者是由于扫描等原因由矩形变成的梯形）
    if len(approx)==4: 
        # 将外轮廓进行倾斜校正，将其构成一个矩形
        # 处理后，仅仅保留答题卡部分，答题卡外面的边界被删除
        # 原始图像的倾斜校正，用于后续标注
        paper = myWarpPerspective(img, approx.reshape(4, 2))
        # cv2.imshow("imgpaper", paper)
        # 原始图像的灰度图像的倾斜校正，用于后续计算
        paperGray = myWarpPerspective(gray, approx.reshape(4, 2))
        # 需要注意，paperGray与paper外观上无差异
        # 但是，paper是彩色空间，可以在上面绘制红色标注信息
        # paperGray是灰度空间
        # cv2.imshow("paper", paper)
        # cv2.imshow("paperGray", paperGray)
        # cv2.imwrite("paperGray.jpg",paperGray)
        # 反二值化阈值处理，选项处理为白色，答题卡整体背景处理黑色
        ret,thresh = cv2.threshold(paperGray, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        # cv2.imshow("thresh", thresh)
        # cv2.imwrite("thresh.jpg",thresh)
        # 在答题纸内寻找所有轮廓，注意此时会找到所有轮廓
        # 既包含各个选项的，还包含答题纸内的文字等内容的轮廓
        # thresh.copy在其copy对象上查找
        cnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # print("找到轮廓个数：",len(cnts))
        # 用options来保存每一个选项（选中，未选中都放进去）
        options = []
        # 遍历每一个轮廓cnts，将选项放入到options中
        # 依据两个条件：
        # 条件1：轮廓如果宽度、高度都大于25像素
        # 条件2：纵横比在[0.6,1.3]之间
        # 同时满足上述两个条件，判定为选项，否则是噪声（文字等其他信息）
        for ci in cnts:
            # 获取轮廓的矩形包围框
            x, y, w, h = cv2.boundingRect(ci)
            #ar纵横比
            ar = w / float(h)
            #满足长度、宽度大于25像素，纵横比在[0.6,1.3]之间，加入到options中
            if w >= 25 and h >= 25 and ar >= 0.6 and ar <= 1.3:
                options.append(ci)
        # print(len(options))  # 看看得到多少个选项的轮廓
        # 得到了很多选项的轮廓，但是他们在options是无规则存放的
        # 将轮廓自上向下存放
        boundingBoxes = [cv2.boundingRect(c) for c in options]
        (options, boundingBoxes) = zip(*sorted(zip(options, boundingBoxes),
                                    key=lambda b: b[1][1], reverse=False))
        #轮廓在options内是自上向下存放的，
        # 因此，options内索引为0、1、2、3的轮廓是第1题的选项轮廓
        # 索引为4、5、6、7的轮廓是第2道题的轮廓，以此类推
        # 简而言之，options内轮廓以步长为4划分，分别对应着不同题目的四个轮廓
        # 从options内，每次取出4个轮廓，分别处理各个题目的各个选项轮廓
        # 使用for循环，从options内，每次取出4个轮廓，处理每一道题的4个选项的轮廓
        # for循环使用tn表示题目序号topic number，i表示轮廓序号（从0开始，步长为4）        
        for (tn, i) in enumerate(np.arange(0, len(options), 4)):
            # 需要注意，取出的4个轮廓，对应某一道题的4个选项
            # 但是这4个选项的存放是无序的
            # 将轮廓按照坐标实现自左向右顺次存放
            # 将选项A、选项B、选项C、选项D，按照坐标顺次存放
            boundingBoxes = [cv2.boundingRect(c) for c in options[i:i + 4]]
            (cnts, boundingBoxes) = zip(*sorted(zip(options[i:i + 4], boundingBoxes),
                                    key=lambda b: b[1][0], reverse=False))
            #构建列表ioptions，用来存储当前题目的每个选项(非零值个数,序号)
            ioptions=[]
            # 使用for循环，提取出4个轮廓的每一个c，及序号ci(contours index)            
            for (ci, c) in enumerate(cnts):
                # 构造一个核答题纸同尺寸的mask，灰度图像，黑色（值均为0）
                mask = np.zeros(paperGray.shape, dtype="uint8")
                # 在mask内，绘制当前遍历到的选项轮廓
                cv2.drawContours(mask, [c], -1, 255, -1)
                # 使用按位与运算的mask模式，提取出当前遍历到的选项
                mask = cv2.bitwise_and(thresh, thresh, mask=mask)
                # cv2.imshow("c" + str(i)+","+str(ci), mask)
                # 计算当前遍历到选项内非零值个数
                total = cv2.countNonZero(mask)
                #将选项非零值个数、选项序号放入列表options内
                ioptions.append((total,ci))
            # 将每道题的4个选项按照非零值个数降序排序
            ioptions=sorted(ioptions,key=lambda x: x[0],reverse=True)
            # 获取包含最多白色像素点的选项索引（序号）
            choiceNum=ioptions[0][1]
            # 根据索引确定选项值：ABCD
            choice=answerDICT.get(choiceNum)
            # print("该生的选项：",choice)
            # 设定标注的颜色类型，绿对红错
            if ANSWER.get(tn) == choiceNum:
                # 正确时，颜色为绿色
                color = (0, 255, 0)  
                # 答对数量加1
                rightSum +=1
            else:
                # 错误时，颜色为红色
                color = (0, 0, 255)   
            cv2.drawContours(paper, cnts[choiceNum], -1, color, 2)
        # cv2.imshow("result", paper)
        s1 = "total: " + str(len(ANSWER)) + ""
        s2 = "right: " + str(rightSum)
        s3 = "score: " + str(rightSum*1.0/len(ANSWER)*100)+""
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(paper, s1 + "  " + s2+"  "+s3, (10, 30), font, 0.5, (0, 0, 255), 2)
        cv2.imshow("score", paper)
        # 不用都遍历了，找到第一个具有4个顶点轮廓，就是答题纸，直接break跳出循环
        break
cv2.waitKey(0)
cv2.destroyAllWindows()
