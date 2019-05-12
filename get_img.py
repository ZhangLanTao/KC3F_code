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
    return bird_s_eye

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
