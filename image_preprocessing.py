import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
import os
from rembg import remove
from PIL import Image
import pilldata as pdt


class Image_Preprocessing():
    def __init__(self, img_path):
        self._image = cv2.imread(img_path)

    def get_cvt_gray(self):
        image_rbg = remove(self._image)

        # 이미지의 대비를 증가시킴
        image_rbg = self.img_Contrast(image_rbg)
        
        # 이미지를 흑백으로 변환
        image_rbg_gray = cv2.cvtColor(image_rbg, cv2.COLOR_BGR2GRAY)

        # 가우시안 블러 처리
        blur = cv2.GaussianBlur(image_rbg_gray, (5, 5), sigmaX=0)
        
        # 밝기와 대비 증가
        enhanced_image = cv2.convertScaleAbs(blur, alpha=2.8, beta=20)

        # 이진화 적용
        _, binary_image = cv2.threshold(enhanced_image, 220, 255, cv2.THRESH_BINARY)  # 이진화
        
        # morphological 변환 적용
        dilated = self.opening(binary_image)

        self._cvt_gray = dilated
        return self._cvt_gray
    
    # kernel과 일치하는 부분에 하나라도 0이 있으면 해당 부분을 모두 채움
    def dilation(self, img):
        # kernel size가 커질수록 영향 커짐
        kernel = np.ones((5,5), np.uint8)
        result = cv2.dilate(img, kernel, iterations=1)
        return result
    
    # kernel과 일치하는 부분에 하나라도 0이 있으면 해당 부분을 모두 제거
    def erosion(self, img):
        # kernel size가 커질수록 영향 커짐
        kernel = np.ones((5,5), np.uint8)
        result = cv2.erode(img, kernel, iterations=1)
        return result
    
    #  erosion 적용 후 dilation 적용, 잡티 제거 효과
    def opening(self, img):
        kernel = np.ones((5,5), np.uint8)
        result = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        return result

    # dilation 적용 후 erosion 적용, 윤곽이 도드라지는 효과
    def closing(self, img):
        kernel = np.ones((5,5), np.uint8)
        result = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        return result
    
    # 이미지의 대비를 증가시킴
    def img_Contrast(self, img):
        # -----Converting image to LAB Color model-----------------------------------
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

        # -----Splitting the LAB image to different channels-------------------------
        l, a, b = cv2.split(lab)

        # -----Applying CLAHE to L-channel-------------------------------------------
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)

        # -----Merge the CLAHE enhanced L-channel with the a and b channel-----------
        limg = cv2.merge((cl, a, b))

        # -----Converting image from LAB Color model to RGB model--------------------
        result = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

        return result

