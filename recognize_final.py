# -*- coding:utf-8 -*-
import numpy as np
from cv2 import *
import math
from get_img import*
import time

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
        return 0
    # 遍历判断
    result = []
    for i in range(len(hash1)):
        # 不相等则n计数+1，n最终为相似度
        result.append(hash1[i] - hash2[i])
    return np.linalg.norm(result)
def get_sign(frame):
    #inrange 过滤蓝色
    HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    cv2.imshow("HSV", HSV)
#此参数难调，尤其是逆光的时候容易崩
    Lower = np.array([110, 80, 40])
    Upper = np.array([130, 255, 155])

    bin = cv2.inRange(HSV, Lower, Upper)
    kernel = np.ones((5, 5), np.uint8)
    bin = cv2.dilate(bin, kernel)

    cv2.imshow("bin", bin)

    #检测边界，拟合椭圆，抠图
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

            print("椭圆信息：", S2, eccentricity, S1/S2)
            cv2.ellipse(frame, ell, (0, 255,), -1)
            #抠图，拉伸
            mask = np.zeros(bin.shape[:2], np.uint8)                  #准备扣标志图的mask
            if (S1 / S2) > 0.85 and S2 > 3000 and S2 < 20000 and eccentricity < 0.5:  # 面积比例，反映拟合情况
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
                cv2.imshow("sign",sign)

                return sign
    return None

#KNN临近算法得到的三个标志的“中心”
left_hash = [0,0,0.014705882,0.073529412,0.191176471,0.411764706,0.808823529,0.338235294,0,0,0.014705882,0.058823529,0,
             0.044117647,0.176470588,0.970588235,0,0.102941176,0.926470588,0.147058824,0.014705882,0,0.014705882,
             0.397058824,0,0.235294118,0.970588235,0.014705882,0.294117647,0.602941176,0,0.147058824,0,0.014705882,
             0.058823529,0.058823529,0.132352941,0.882352941,0,0.102941176,0.014705882,0,0.029411765,0.014705882,
             0.117647059,0.970588235,0,0.382352941,0,0.014705882,0,0,0.014705882,0.014705882,0.176470588,0.955882353,
             0,0,0,0.058823529,0.088235294,0.441176471,0.867647059,0.411764706]

straight_hash = [0,0,0,0,0,0.838120104,0.736292428,0,0,0,0.18537859,0.916449086,0,0,0.01305483,1,0,0.015665796,
                 0.997389034,0,0,0,0.036553525,0.548302872,0,0.002610966,0.002610966,1,0,0.005221932,0.033942559,
                 0.266318538,0,0,0.002610966,1,0,0.002610966,0.101827676,0.219321149,0,0,0.002610966,1,0,0.020887728,
                 0.146214099,0.459530026,0,0,0.070496084,0.924281984,0.018276762,0.060052219,0.255874674,0.830287206,
                 0,0,0,0.015665796,0.078328982,0.772845953,0.650130548,0]
right_hash = [0,0,0.013888889,0.138888889,0.180555556,0.458333333,0.819444444,0.388888889,0,0,0,0.055555556,0.027777778,
              0,0.208333333,0.930555556,0,0.083333333,0.361111111,0.652777778,0.791666667,0,0,0.458333333,0.027777778,
              0.916666667,0.277777778,0.486111111,0.694444444,0,0.027777778,0.180555556,0.027777778,0.972222222,
              0.013888889,0.055555556,0.055555556,0,0,0.111111111,0.041666667,0.972222222,0.013888889,0,0,0.013888889,
              0.013888889,0.402777778,0,0.069444444,0,0,0.027777778,0,0.125,0.958333333,0,0,0,0.055555556,0.138888889,
              0.416666667,0.875,0.319444444]

def recognize_sign(sign):
    if sign is not None:
        cv2.imshow("sign", sign)
        sign_hash = dHash(sign)
        list = []       #顺序：左转 直走 右转
        list.append(cmpHash(left_hash, sign_hash))
        list.append(cmpHash(straight_hash, sign_hash))
        list.append(cmpHash(right_hash, sign_hash))
        print (list)
        max_index = list.index(min(list))
        print ("max match",max_index)
        return max_index
    else:
        print('no sign')
        return -1

def match_test(filedir):
    # 以下代码检查一个文件夹里的所有图片的匹配情况
    filelist = cv2.os.listdir(filedir)  # 列出文件夹下所有的目录与文件
    for i in range(0, len(filelist)):
        path = cv2.os.path.join(filedir, filelist[i])
        if cv2.os.path.isfile(path):
            frame = get_img_from_file(path, CameraParameter)
            sign = get_sign(frame)
            rec_result = recognize_sign(sign)
            cv2.imshow("frame", frame)
            if rec_result == -1:
                c = cv2.waitKey(20)
                while (c != 99):
                    c = cv2.waitKey(20)
            cv2.waitKey(20)

if __name__ == "__main__":
    Camera, CameraParameter = camera_parameters_init()
    #filedir = "./realimg/polarizer_bright/right"
    #match_test(filedir)
    #frame = get_img_from_file("./realimg/polarizer_bright/right/WIN_20190519_15_28_44_Pro (2).jpg", CameraParameter)
    while 1:
        time0 = time.process_time()
        frame = get_img_from_camera(Camera, CameraParameter, False)     #False：巡线时不去畸，否则很慢
        bird = get_bird_img(frame, CameraParameter)
        time1 = time.process_time()
        dt = time1-time0
        if dt<0.001:
            dt = 0.001
        freq = 1/dt
        print("frequency:",freq)
        imshow("bird", bird)
        imshow("frame", frame)

        waitKey(20)
'''        
    frame = get_img_from_file("./realimg/polarizer_bright/right/WIN_20190519_15_28_44_Pro (2).jpg", CameraParameter)
    #frame = get_img_from_camera(Camera, CameraParameter)
    sign = get_sign(frame)
    recognize_sign(sign)
    cv2.imshow("frame", frame)
'''
        #bird_s_eye = get_bird_img(frame, CameraParameter)
        #road = skeletonize(bird_s_eye)
        #cv2.imshow("road", road)

        #kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 9))
        #road_x = cv2.morphologyEx(road, cv2.MORPH_DILATE, kernel)
        #cv2.imshow("road_x",road_x)

        #查表法细化
        #thin = to_thin(bird_s_eye)
        #cv2.imshow("thin",thin)

        #frame = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
        #edges = cv2.Canny(frame, 50, 150, apertureSize=3)
        #cv2.imshow("edge", edges)

        #found_sign = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 100)
        #found_sign = np.uint16(np.around(found_sign))
        #for i in found_sign[0, :]:
        #    cv2.circle(frame, (i[0], i[1]), i[2], (0, 0, 0), 2)
        #    cv2.circle(frame, (i[0], i[1]), 5, (0, 0, 0), 2)

        #display_camera(Camera, CameraParameter)
        #cv2.waitKey(20)
