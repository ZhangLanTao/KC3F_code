import cv2
import numpy as np
import math

def get_carline(img):
    pass
"""
使用Opencv Sobel算子来求方向梯度
	img: Grayscale
	direction : x or y axis
	thresh : apply threshold on pixel intensity of gradient image
	output is binary image
"""
def directional_gradient(img):
    ddepth = cv2.CV_64F
    scale = 1
    delta = 0
    # 高斯模糊
    cv2.GaussianBlur(img,img,(3,3),0,0,cv2.BORDER_DEFAULT)
    
    # 转成灰度图
    cv2.cvtColor(img, img_gray, cv2.RGB2GRAY)

    # 进行微分求两个方向上的导数
    cv2.Sobel(img_gray, grad_x, ddepth, 1, 0, 3, scale, delta, CV2_BORDER_DEFAULT)
    cv2.convertScaleAbs(grad_x, abs_grad_x)

    cv2.Sobel(img_gray, grad_y, ddepth, 0, 1, 3, scale, delta, CV2_BORDER_DEFAULT)
    cv2.convertScaleAbs(grad_y, abs_grad_y)

    cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad)

    img = grad
    print("1")
    cv2.imshow(img)
    cv2.WaitKey()

def Sobel(img):
    sobelx=cv2.Sobel(img,cv2.CV_64F,dx=1,dy=0) #x方向的
    #使cv2.convertScaleAbs()函数将结果转化为原来的uint8的形式
    sobelx=cv2.convertScaleAbs(sobelx)
    
    sobely=cv2.Sobel(img,cv2.CV_64F,dx=0,dy=1) #y方向的
    sobely=cv2.convertScaleAbs(sobely)
    
    result=cv2.addWeighted(sobelx,0.5,sobely,0.5,0)
    #x方向和y方向的梯度权重
    # cv2.imshow("Original",img)
    # cv2.imshow("sobelx",sobelx)
    # cv2.imshow("sobely",sobely)
    # cv2.imshow("result",result)
    return result
"""
    cv2.waitKey()
    cv2.destroyAllWindows()
"""
if __name__ == "__main__":
    img = cv2.imread("img1.jpg", cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = Sobel(img)  # 梯度算法
    ret, result = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
    cv2.imshow("result",result)
    cv2.waitKey()
    cv2.destroyAllWindows()
