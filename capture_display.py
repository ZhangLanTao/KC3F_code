# -*- coding:utf-8 -*-
#从摄像头获得图像并显示。
import numpy as np
import cv2

def camera_parameters_init():
    CameraMtx = np.array([[920.341540386842, 0.928045375071153, 638.842904534224], [0, 919.186342935789, 360.592128174277], [0, 0, 1]])
    DistCoeffs = np.array([[-0.399507111526893], [0.166350759592159], [0.000800934404473862], [0.000814484527629980], [0]])
    return CameraMtx , DistCoeffs

def display():
    CameraMtx ,DistCoeffs = camera_parameters_init()
    cap = cv2.VideoCapture(1)#cap仅仅是摄像头的一个对象
    while True:
        ret,frame = cap.read()#一帧一帧的捕获视频,ret返回的是否读取成功，frame返回的是帧
        frame = np.rot90(frame)
        frame = np.rot90(frame)
        cv2.imshow('raw', frame)

        frame = cv2.undistort(frame, CameraMtx ,DistCoeffs)
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    display()