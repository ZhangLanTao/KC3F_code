# -*- coding:utf-8 -*-
import numpy as np
from cv2 import *
import math
from get_img import *


def rotate(image, angle, center=None, scale=1.0):
    (h, w) = image.shape[:2]
    if center is None:
        center = (w // 2, h // 2)
    M = getRotationMatrix2D(center, angle, scale)
    rotated = warpAffine(image, M, (w, h))
    return rotated


def dHash_to_file(img, file):
    if (img is None):
        print("NO SIGN")
        return

    # 缩放8*8
    img = resize(img, (9, 8), interpolation=INTER_CUBIC)
    # 转换灰度图
    # gray = cvtColor(img, COLOR_BGR2GRAY)
    gray = img
    for i in range(8):
        for j in range(8):
            if gray[i, j] > gray[i, j + 1]:
                file.write("1\t")
            else:
                file.write("0\t")
    file.write("\n")


def get_sign(frame):
    if frame is None:
        print("no image for function get_sign.")
        return -1
    # inrange 过滤蓝色
    HSV = cvtColor(frame, COLOR_BGR2HSV)

    imshow("HSV", HSV)

    Lower = np.array([110, 150, 40])
    Upper = np.array([130, 255, 155])

    bin = inRange(HSV, Lower, Upper)
    kernel = np.ones((5, 5), np.uint8)
    bin = dilate(bin, kernel)

    imshow("bin", bin)

    # 检测边界，拟合椭圆，抠图
    contours, hierarchy = findContours(bin, RETR_EXTERNAL,CHAIN_APPROX_SIMPLE)  # contours为轮廓集，可以计算轮廓的长度、面积等
    for cnt in contours:
        if len(cnt) > 50:
            # 拟合椭圆信息
            ell = fitEllipse(cnt)
            b2, a2 = ell[1][:2]
            a = a2 / 2
            b = b2 / 2
            S1 = contourArea(cnt)
            S2 = math.pi * a * b
            eccentricity = math.sqrt(a * a - b * b) / a

            print("椭圆信息：", S2, eccentricity, S1 / S2)
            ellipse(frame, ell, (0, 255, ), -1)
            # 抠图，拉伸
            mask = np.zeros(bin.shape[:2], np.uint8)  # 准备扣标志图的mask
            if (S1 / S2) > 0.85 and S2 > 2000 and S2 < 20000 and eccentricity < 0.6:  # 面积比例，反映拟合情况
                ellipse(mask, ell, (255, 255, 255), -1)
                sign = bitwise_and(bin, bin, mask=mask)  # 标志ROI
                x, y, w, h = boundingRect(cnt)  # 裁标志
                rate = ell[1][1] / ell[1][0]  # 拉伸比例
                angle = ell[2]  # 旋转角度
                sign = sign[y:y + h, x:x + w]  # 原图标志部分裁出

                sign = rotate(sign, angle, scale=0.8)  # 旋转，拉伸，旋转回去
                sign = resize(sign, None, fx=rate, fy=1)
                sign = rotate(sign, -angle, scale=1.125)
                h, w = sign.shape[0:2]

                sign = sign[5:h - 5, (w - h) // 2 + 5:(w + h) // 2 - 5]  # 变成正方形
                destroyWindow("sign")
                imshow("sign", sign)
                return sign
    return None


# KNN临近算法得到的三个标志的“中心” 待定
left_hash = []
straight_hash = []
right_hash = []

if __name__ == "__main__":
    Camera, CameraParameter = camera_parameters_init()
    dstfile = open("right.txt", 'w')
    filedir = "../../realimg/polarizer_bright/right"

    filelist = os.listdir(filedir)  # 列出文件夹下所有的目录与文件
    for i in range(0, len(filelist)):
        path = os.path.join(filedir, filelist[i])
        if os.path.isfile(path):
            frame = get_img_from_file(path, CameraParameter)
            sign = get_sign(frame)
            dHash_to_file(sign, dstfile)

            imshow("frame", frame)
            if sign is None:
                print("此路标不能识别，需要更改阈值：", path)
                c = waitKey(20)
                while ( c != 99):
                    c = waitKey(20)

            waitKey(20)
    dstfile.close()
