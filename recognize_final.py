# -*- coding:utf-8 -*-
import numpy as np
import cv2
import math
from get_img import*

def rotate(image, angle, center=None, scale=1.0):
    (h, w) = image.shape[:2]
    if center is None:
        center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

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

'''def dHash(img):
    # 缩放8*8
    img = cv2.resize(img, (9, 8), interpolation=cv2.INTER_CUBIC)
    # 转换灰度图
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = img
    hash_str = ''
    # 每行前一个像素大于后一个像素为1，相反为0，生成哈希
    for i in range(8):
        for j in range(8):
            if gray[i, j] > gray[i, j + 1]:
                hash_str = hash_str + '1' +'\t'
            else:
                hash_str = hash_str + '0' +'\t'
    return hash_str'''
def dHash(img):
    # 缩放8*8
    img = cv2.resize(img, (9, 8), interpolation=cv2.INTER_CUBIC)
    # 转换灰度图
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = img
    hash_str = []    # 每行前一个像素大于后一个像素为1，相反为0，生成哈希
    for i in range(8):
        for j in range(8):
            if gray[i, j] > gray[i, j + 1]:
                hash_str.append(1)
            else:
                hash_str.append(0)
    return hash_str

'''def cmpHash(hash1, hash2):
    n = 0
    # hash长度不同则返回-1代表传参出错
    if len(hash1) != len(hash2):
        return -1
    # 遍历判断
    for i in range(len(hash1)):
        # 不相等则n计数+1，n最终为相似度
        if hash1[i] != hash2[i]:
            n = n + 1
    return 1 - n / 64'''
def cmpHash(hash1, hash2):
    n = 0
    # hash长度不同则返回-1代表传参出错
    if len(hash1) != len(hash2):
        return -1
    # 遍历判断
    result = []
    for i in range(len(hash1)):
        # 不相等则n计数+1，n最终为相似度
        result.append(hash1[i] - hash2[i])
    return np.linalg.norm(result)
def get_sign(frame):
    HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    Lower = np.array([90, 100, 170])
    Upper = np.array([120, 255, 255])
    '''此效果较好，勿删
    Lower = np.array([90, 100, 170])
    Upper = np.array([120, 255, 255])'''
    bin = cv2.inRange(HSV, Lower, Upper)
    kernel = np.ones((5, 5), np.uint8)
    bin = cv2.dilate(bin, kernel)
    cv2.imshow("bin", bin)
    contours, hierarchy = cv2.findContours(bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # contours为轮廓集，可以计算轮廓的长度、面积等
    for cnt in contours:
        if len(cnt) > 50:
            #拟合椭圆信息
            ell = cv2.fitEllipse(cnt)
            b2, a2 = ell[1][:2]
            a = a2 / 2
            b = b2 / 2
            S1 = cv2.contourArea(cnt)
            S2 = math.pi * a * b
            eccentricity = math.sqrt(a * a - b * b) / a

            #抠图，拉伸
            mask = np.zeros(bin.shape[:2], np.uint8)                  #准备扣标志图的mask
            if (S1 / S2) > 0.85 and S2 > 10000 and eccentricity < 0.5:  # 面积比例，反映拟合情况
                cv2.ellipse(mask, ell, (255, 255, 255), -1)
                sign = cv2.bitwise_and(bin, bin, mask=mask)         #标志ROI
                x, y, w, h = cv2.boundingRect(cnt)                      #裁标志
                rate = ell[1][1] / ell[1][0]                            #拉伸比例
                angle = ell[2]                                          #旋转角度
                sign = sign[y:y + h, x:x + w]                           #原图标志部分裁出

                sign = rotate(sign, angle, scale=0.8)                   #旋转，拉伸，旋转回去
                sign = cv2.resize(sign, None, fx=rate, fy=1)
                sign = rotate(sign, -angle, scale=1.125)
                h, w = sign.shape[0:2]

                sign = sign[5:h-5, (w - h) // 2 + 5:(w + h) // 2 -5]            #变成正方形
                return sign
    return None
'''
right_hash = '1100010010000110101000110001101110010110100101111001110001000100'
straight_hash = '0110100010101100100011010010100000001100000010011010100001001000'
left_hash = '1110000010100000101001010001010110010100001001011000010001000100'
'''
left_hash = [0,0,0,0,0,0.407158837,0.995525727,0.051454139,0,0.006711409,0.313199105,0,0,0,0.006711409,1,0.163310962,
             0.991051454,0,0.004474273,0.006711409,0,0,0.208053691,0.25950783,0.850111857,0.017897092,0.601789709,
             0.991051454,0,0.002237136,0.239373602,0,0.080536913,0.315436242,0,1,0.002237136,0.029082774,
             0.248322148,0,0,0,0,0.977628635,0.015659955,0.031319911,0.326621924,0,0,0,0.071588367,0.836689038,
             0.002237136,0.129753915,0.953020134,0,0,0,0,0.002237136,0.362416107,0.988814318,0.080536913]
straight_hash = [0,0,0,0,0,0.838120104,0.736292428,0,0,0,0.18537859,0.916449086,0,0,0.01305483,1,0,0.015665796,
                 0.997389034,0,0,0,0.036553525,0.548302872,0,0.002610966,0.002610966,1,0,0.005221932,0.033942559,
                 0.266318538,0,0,0.002610966,1,0,0.002610966,0.101827676,0.219321149,0,0,0.002610966,1,0,0.020887728,
                 0.146214099,0.459530026,0,0,0.070496084,0.924281984,0.018276762,0.060052219,0.255874674,0.830287206,
                 0,0,0,0.015665796,0.078328982,0.772845953,0.650130548,0]
right_hash = [0,0,0,0,0.006012024,0.811623246,0.829659319,0,0,0,0,0,0.226452906,0.014028056,0.018036072,0.995991984,0,
              0,0.008016032,0.026052104,0.274549098,0.873747495,0,0.438877756,0,0,1,0,0.488977956,0.883767535,0,
              0.036072144,0,0,1,0,0.206412826,0.006012024,0.056112224,0.078156313,0,0,0.995991984,0.002004008,0,
              0.006012024,0.054108216,0.440881764,0,0,0.785571142,0.178356713,0,0.002004008,0.094188377,0.991983968,
              0,0,0,0,0,0.841683367,0.80761523,0.002004008]
def recognize_sign(sign):
    if sign is not None:
        sign_hash = dHash(sign)
        list = []       #顺序：左转 直走 右转
        list.append(cmpHash(left_hash, sign_hash))
        list.append(cmpHash(straight_hash, sign_hash))
        list.append(cmpHash(right_hash, sign_hash))

        max_index = list.index(min(list))
        print ("max match",max_index)
    else:
        print('no sign')

if __name__ == "__main__":
    Camera, CameraParameter = camera_parameters_init()
    while 1:
        #frame = get_img_from_file("2.jpg", CameraParameter)
        frame = get_img_from_camera(Camera, CameraParameter)
        sign = get_sign(frame)
        #cv2.imshow("sign", sign)
        #recognize_sign(sign)
        if sign is not None:
            recognize_sign(sign)
            cv2.imshow("sign",sign)
        else:
            print("no sign")
        cv2.imshow("frame", frame)

        c = cv2.waitKey(1)
        if c == 99:
            break
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
