import cv2
import numpy as np

def show_img(img):
    cv2.imshow("img", img)
    cv2.waitKey(0)

img=cv2.imread("./resources/car.jpg",)
img=cv2.resize(img,(1024,800))

gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#双边滤波
#第一个参数  输入图像
#第二个参数  领域直径
#第三个参数  颜色空间的标准差
#第四个参数  坐标空间的标准差
blf=cv2.bilateralFilter(gray,13,15,15)

show_img(blf)

#边缘检测
edge=cv2.Canny(blf,30,200)

#寻找轮廓
contours,_=cv2.findContours(edge.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

#最大前10个 轮廓面积排序 reverse 从大到小
contours=sorted(contours, key=cv2.contourArea, reverse=True)[:10]

