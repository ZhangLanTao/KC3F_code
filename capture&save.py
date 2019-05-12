# -*- coding:utf-8 -*-

import numpy as np
import cv2
import math



def save_img_from_camera(Camera,filename):   #return 去畸后的一张图片
    ret, frame = Camera.read() # 一帧一帧的捕获视频,ret返回的是否读取成功，frame返回的是帧
    frame = np.rot90(frame)
    frame = np.rot90(frame)
    cv2.imshow("frame", frame)
    cv2.imwrite(str(filename)+".jpg",frame)

if __name__ == "__main__":
    Camera = cv2.VideoCapture(0)  # cap仅仅是摄像头的一个对象
    i = 1
    while 1:
        save_img_from_camera(Camera,i)
        i += 1
        while cv2.waitKey(2) != ord('n'):

