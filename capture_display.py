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
    Camera = cv2.VideoCapture(1)   #cap仅仅是摄像头的一个对象
    Camera.set(3, 1280)             #设置分辨率
    Camera.set(4, 720)
    #内参外参由matlab标定,第一次
    CameraMtx = np.array([[920.341540386842, 0.928045375071153, 638.842904534224], [0, 919.186342935789, 360.592128174277], [0, 0, 1]])
    DistCoeffs = np.array([[-0.399507111526893], [0.166350759592159], [0.000800934404473862], [0.000814484527629980], [0]])

    #CameraMtx = np.array([[919.291833890046, -3.54778267427631, 625.722019056929], [0, 920.660382300987, 365.959618825537], [0, 0, 1]])
    #DistCoeffs = np.array([[-0.367827834522785], [0.121845281585466], [0.121845281585466], [0.00223734605257760], [0]])

    #2阶 暂可
    #CameraMtx = np.array([[919.291833890046, -3.54778267427631, 625.722019056929], [0, 920.660382300987, 365.959618825537], [0, 0, 1]])
    #DistCoeffs = np.array([[-0.367827834522785], [0.121845281585466], [-7.75756941045992e-05], [0.00223734605257760], [0]])
    #暂可 3阶
    #CameraMtx = np.array([[914.917957805213, -3.22272338797268, 636.147831578619], [0, 915.810601312156, 370.745004939807], [0, 0, 1]])
    #DistCoeffs = np.array([[-0.402720345573782], [0.222929378250051], [-0.0756545687359382], [0.00220619282231953], [-0.0756545687359382]])

    #透视变换原图四点和期望的四点位置
    #计算透视变换矩阵
    src_points = np.array([[73., 713.], [1195., 697.], [277., 501.], [992., 487.]], dtype="float32")
    dst_points = np.array([[110., 260.], [290., 260.], [0., 0.], [400., 0.]], dtype="float32")
    TransferM = cv2.getPerspectiveTransform(src_points, dst_points)
    CameraParameter = CameraParameters(CameraMtx , DistCoeffs ,TransferM)
    return Camera , CameraParameter

#return 去畸后的图片和鸟瞰图
#透视变换，变成鸟瞰图
"""void cv::warpPerspective(
		cv::InputArray src, // 输入图像
		cv::OutputArray dst, // 输出图像
		cv::InputArray M, // 3x3 变换矩阵
		cv::Size dsize, // 目标图像大小
		int flags = cv::INTER_LINEAR, // 插值方法
		int borderMode = cv::BORDER_CONSTANT, // 外推方法
		const cv::Scalar& borderValue = cv::Scalar() //常量边界时使用
	);"""
def get_img_from_camera(Camera,CameraParameter):
    ret, frame = Camera.read()  # 一帧一帧的捕获视频,ret返回的是否读取成功，frame返回的是帧
    frame = cv2.undistort(frame, CameraParameter.CameraMtx, CameraParameter.DistCoeffs)
    frame = np.rot90(frame)
    frame = np.rot90(frame)

    bird_s_eye = get_bird_img(frame,CameraParameter)
    return frame,bird_s_eye

def get_img_from_file(file_name,CameraParameter):
    frame = cv2.imread(file_name, cv2.IMREAD_UNCHANGED)
    frame = cv2.undistort(frame, CameraParameter.CameraMtx, CameraParameter.DistCoeffs)
    return frame
def find_road(img):
    #将x y方向梯度叠加，腐蚀 增强路面图像
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ret, img_gray = cv2.threshold(img_gray, 120, 255, cv2.THRESH_BINARY)
    #img_gray = 255 - img_gray
    cv2.imshow("test", img_gray)
    sobelx = cv2.Sobel(img_gray, cv2.CV_64F, dx=1, dy=0)  # x方向的
    sobelx = cv2.convertScaleAbs(sobelx)
    sobelx = 255 - sobelx
    sobely = cv2.Sobel(img_gray, cv2.CV_64F, dx=0, dy=1)  # y方向的
    sobely = cv2.convertScaleAbs(sobely)
    sobely = 255-sobely
    road_enhanced = sobelx & sobely
    ret , road_enhanced = cv2.threshold(road_enhanced,100,255,cv2.THRESH_BINARY)

    #kernal = cv2.getStructuringElement(cv2.MORPH_RECT,(9,7))
    #road_enhanced = cv2.erode(road_enhanced,kernal)
    #road_enhanced = road_enhanced
    #road_enhanced = cv2.dilate(road_enhanced,kernal)
    #kernal = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 3))
    #road_enhanced = cv2.dilate(road_enhanced, kernal)

    #cv2.imshow("x", sobelx)
    #cv2.imshow("y", sobely)
    #cv2.imshow("enhanced", road_enhanced)

def display_camera(Camera, CameraParameter):        #从camera读一张图，去畸，鸟瞰
    while True:
        frame, bird_s_eye = get_img_from_camera(Camera, CameraParameter)
        cv2.imshow('frame', frame)
        cv2.imshow('bird“s eye', bird_s_eye)
        if cv2.waitKey(1) == ord('q'):
            break
    Camera.release()
    cv2.destroyAllWindows()

#def find_road_sign(src):



if __name__ == "__main__":
    Camera, CameraParameter = camera_parameters_init()
    while 1:
        frame = get_img_from_file("1.jpg", CameraParameter)
        cv2.imshow("frame", frame)
        HLS = cv2.cvtColor(frame, cv2.COLOR_RGB2HLS)
        #cv2.imshow("HLS", HLS)

        HLS = cv2.pyrMeanShiftFiltering(HLS, 10, 10)
        #cv2.imshow("HLS_F", HLS)

        #对H通道模糊、二值化，找蓝色区域在哪里
        HLS_split = cv2.split(HLS)
        HLS_split_blur = cv2.blur(HLS_split[2], (7,5))
        ret, HLS_split_bin = cv2.threshold(HLS_split_blur, 245, 255, cv2.THRESH_BINARY)

        #开运算去除水波纹导致的小的白色区域
        kernel = np.ones((7, 9), np.uint8)
        HLS_split_bin_opening = cv2.morphologyEx(HLS_split_bin, cv2.MORPH_OPEN, kernel)



        contours, hierarchy = cv2.findContours(HLS_split_bin_opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # contours为轮廓集，可以计算轮廓的长度、面积等
        for cnt in contours:
            if len(cnt) > 50:
                S1 = cv2.contourArea(cnt)
                ell = cv2.fitEllipse(cnt)
                S2 = math.pi * ell[1][0] * ell[1][1]/4
                if (S1 / S2) > 0.9:  # 面积比例，反映拟合情况
                    print(S1/S2)
                    cv2.ellipse(frame, ell, (0, 0, 255), 2)
        cv2.imshow("frame", frame)
        #bird_s_eye = get_bird_img(frame, CameraParameter)
        #cv2.imshow("bird", bird_s_eye)
        #find_road(bird_s_eye)
        #frame = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
        #edges = cv2.Canny(frame, 50, 150, apertureSize=3)
        #cv2.imshow("edge", edges)

        #found_sign = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 100)
        #found_sign = np.uint16(np.around(found_sign))
        #for i in found_sign[0, :]:
        #    cv2.circle(frame, (i[0], i[1]), i[2], (0, 0, 0), 2)
        #    cv2.circle(frame, (i[0], i[1]), 5, (0, 0, 0), 2)

        #display_camera(Camera, CameraParameter)
        #cv2.waitKey(50)
        cv2.waitKey(0)
