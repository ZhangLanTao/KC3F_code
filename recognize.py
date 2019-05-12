import cv2
import numpy as np
from capture_display import*


def run_car(sm, st):
    print("==========run_car Start==========")
    d = driver()
    try:
        d.setStatus(motor=sm, servo=st, mode="speed")
        print("Motor: %0.2f, Servo: %0.2f" % (sm, st))
    except KeyboardInterrupt:
        pass

    d.close()
    del d
    return 0


def get_sign(img):
    HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    cv2.imshow("test",HSV)
    Lower = np.array([88, 69, 205])
    Upper = np.array([119, 226, 255])
    mask = cv2.inRange(HSV, Lower, Upper)
    wanted = cv2.bitwise_and(img, img, mask=mask)
    gray = cv2.cvtColor(wanted, cv2.COLOR_BGR2GRAY)
    cv2.bitwise_not(gray, gray)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 100, param1=100, param2=30, minRadius=40, maxRadius=200)

    if circles is not None:
        x, y, radius = circles[0][0]
        sign = img[int(y - radius):int(y + radius), int(x - radius):int(x + radius)]
        return sign
    else:
        return False


def dHash(img):
    # 缩放8*8
    img = cv2.resize(img, (9, 8), interpolation=cv2.INTER_CUBIC)
    # 转换灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hash_str = ''
    # 每行前一个像素大于后一个像素为1，相反为0，生成哈希
    for i in range(8):
        for j in range(8):
            if gray[i, j] > gray[i, j + 1]:
                hash_str = hash_str + '1'
            else:
                hash_str = hash_str + '0'
    return hash_str


def cmpHash(hash1, hash2):
    n = 0
    # hash长度不同则返回-1代表传参出错
    if len(hash1) != len(hash2):
        return -1
    # 遍历判断
    for i in range(len(hash1)):
        # 不相等则n计数+1，n最终为相似度
        if hash1[i] != hash2[i]:
            n = n + 1
    return 1 - n / 64


right_hash = '1100010010000110101000110001101110010110100101111001110001000100'
straight_hash = '0110100010101100100011010010100000001100000010011010100001001000'
left_hash = '1110000010100000101001010001010110010100001001011000010001000100'

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    while True:
        #ret, frame = cap.read()
        #cv2.imshow('Video', frame)
        Camera, CameraParameter = camera_parameters_init()
        frame = get_img_from_file("5.jpg", CameraParameter)

        sign = get_sign(frame)
        if sign is not False:
            cv2.imshow('sign',sign)
            sign_hash = dHash(sign)

            r_similar = cmpHash(right_hash, sign_hash)
            s_similar = cmpHash(straight_hash, sign_hash)
            l_similar = cmpHash(left_hash, sign_hash)

            if r_similar > s_similar:
                result = 'right'
                if r_similar > l_similar:
                    result = 'right'
                    #run_car(0.3, 0.5)
                else:
                    result = 'left'
                    #run_car(0.3, -0.5)
            else:
                result = 'straight'
                if s_similar > l_similar:
                    result = 'straight'
                    #run_car(0.3, 0)
                else:
                    result = 'left'
                    #run_car(0.3, -0.5)
            print(result)

        else:
            print('no circle')
        c = cv2.waitKey(1)
        if c == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

