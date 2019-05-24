# -*- coding:utf-8 -*-

import numpy as np
import cv2
import math

class CameraParameters:
    def __init__(self,CameraMtx , DistCoeffs ,TransferM):
        self.CameraMtx = CameraMtx
        self.DistCoeffs = DistCoeffs
        self.TransferM = TransferM

def get_bird_img(src,CameraParameter):
    bird_s_eye = cv2.warpPerspective(src, CameraParameter.TransferM, (400, 260), cv2.INTER_LINEAR)
    for y in range(100):
        for x in range(y):
            bird_s_eye[y + 160, x] = 255
    for y in range(100):
        for x in range(y):
            bird_s_eye[y + 160, 399 - x] = 255
    return bird_s_eye

#提取路径的骨架
#可以修改kernel的样式
#提取二值化图像的黑色的骨架
def skeletonize(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ret, img = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)
    img = cv2.bitwise_not(img)
    skel = img.copy()

    skel[:,:] = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))

    while True:
        eroded = cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel)
        temp = cv2.morphologyEx(eroded, cv2.MORPH_DILATE, kernel)
        temp  = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img[:,:] = eroded[:,:]
        if cv2.countNonZero(img) == 0:
            break
    skel = cv2.bitwise_not(skel)
    skel = cv2.GaussianBlur(skel,(7,7),0,0)
    ret, skel = cv2.threshold(skel, 240, 255, cv2.THRESH_BINARY)
    return skel



# 查表法细化
def to_thin(img):
    array = [0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, \
             1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, \
             0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, \
             1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, \
             1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
             1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, \
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
             0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, \
             1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, \
             0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, \
             1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, \
             1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
             1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, \
             1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, \
             1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ret, img = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)
    cv2.imshow("lmg", img)
    i_thin = img
    for h in range(height):
        for w in range(width):
            if img[h, w] == 0:
                a = [1] * 9
                for i in range(3):
                    for j in range(3):
                        if -1 < h-1+i < height and -1 < w-1+j < width and i_thin[h-1+i, w-1+j] == 0:
                            a[j*3+i] = 0
                sum = a[0]*1 + a[1]*2 + a[2]*4 + a[3]*8 + a[5]*16 + a[6]*32 + a[7]*64 + a[8]*128
                i_thin[h, w] = array[sum] * 255
    return i_thin

def camera_parameters_init():       #return 摄像头对象 摄像头参数    return Camera , CameraMtx , DistCoeffs ,TransferM
    Camera = cv2.VideoCapture(0)   #cap仅仅是摄像头的一个对象
    Camera.set(3, 1280)             #设置分辨率
    Camera.set(4, 720)

    #内参外参由matlab标定
    CameraMtx = np.array([[920.341540386842, 0.928045375071153, 638.842904534224], [0, 919.186342935789, 360.592128174277], [0, 0, 1]])
    DistCoeffs = np.array([[-0.399507111526893], [0.166350759592159], [0.000800934404473862], [0.000814484527629980], [0]])

    #透视变换原图四点和期望的四点位置
    #计算透视变换矩阵
    src_points = np.array([[73., 713.], [1195., 697.], [277., 501.], [992., 487.]], dtype="float32")
    dst_points = np.array([[110., 260.], [290., 260.], [0., 0.], [400., 0.]], dtype="float32")
    TransferM = cv2.getPerspectiveTransform(src_points, dst_points)
    CameraParameter = CameraParameters(CameraMtx , DistCoeffs ,TransferM)
    return Camera , CameraParameter

def get_img_from_camera(Camera,CameraParameter):
    ret, frame = Camera.read()  # 一帧一帧的捕获视频,ret返回的是否读取成功，frame返回的是帧
    frame = cv2.undistort(frame, CameraParameter.CameraMtx, CameraParameter.DistCoeffs)
    frame = np.rot90(frame)
    frame = np.rot90(frame)
    return frame

def get_img_from_file(file_name,CameraParameter):
    frame = cv2.imread(file_name, cv2.IMREAD_UNCHANGED)
    if frame is None:
        print("no such image(get_img_from_file)")
        return None
    frame = np.rot90(frame)
    frame = np.rot90(frame)
    frame = cv2.undistort(frame, CameraParameter.CameraMtx, CameraParameter.DistCoeffs)
    return frame

def get_sign(frame):
    HLS = cv2.cvtColor(frame, cv2.COLOR_RGB2HLS)
    HLS = cv2.pyrMeanShiftFiltering(HLS, 10, 10)

    #对H通道模糊、二值化，找蓝色区域在哪里
    HLS_split = cv2.split(HLS)
    HLS_split_blur = cv2.blur(HLS_split[2], (7,5))
    ret, HLS_split_bin = cv2.threshold(HLS_split_blur, 245, 255, cv2.THRESH_BINARY)

    #开运算去除水波纹导致的小的白色区域
    kernel = np.ones((7, 9), np.uint8)
    HLS_split_bin_opening = cv2.morphologyEx(HLS_split_bin, cv2.MORPH_OPEN, kernel)

    #找边界
    contours, hierarchy = cv2.findContours(HLS_split_bin_opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # contours为轮廓集，可以计算轮廓的长度、面积等
    for cnt in contours:
        if len(cnt) > 50:
            S1 = cv2.contourArea(cnt) #原面积
            ell = cv2.fitEllipse(cnt) #拟合椭圆的面积
            S2 = math.pi * ell[1][0] * ell[1][1]/4
            if (S1 / S2) > 0.9:  # 面积比例，反映拟合情况
                cv2.ellipse(frame, ell, (0, 0, 255), 2)
    cv2.imshow("frame", frame)

if __name__ == "__main__":
    print("no functions to run.")
