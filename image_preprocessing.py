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
        # 이미지 배경 제거
        self._cvt_text_rec = remove(self._image)

        # 이미지의 대비를 증가시킴
        self._cvt_text_rec = self.img_Contrast(self._cvt_text_rec)

        # 이미지를 흑백으로 변환
        self._cvt_text_rec = cv2.cvtColor(self._cvt_text_rec, cv2.COLOR_BGR2GRAY)

        # 가우시안 블러 처리
        self._cvt_text_rec = cv2.GaussianBlur(self._cvt_text_rec, (5, 5), sigmaX=0)

        # 밝기와 대비 증가
        self._cvt_text_rec = cv2.convertScaleAbs(self._cvt_text_rec, alpha=2.8, beta=20)

        # 이진화 적용
        _, self._cvt_text_rec = cv2.threshold(self._cvt_text_rec, 220, 255, cv2.THRESH_BINARY)

        # 이진화된 이미지 흑백 반전
        self._cvt_text_rec = cv2.bitwise_not(self._cvt_text_rec)

        # morphological 변환 적용
        self._cvt_text_rec = self.dilation(self._cvt_text_rec, (9,7))
        self._cvt_text_rec = self.closing(self._cvt_text_rec, (5,5))
        self._cvt_text_rec = self.opening(self._cvt_text_rec, (5,5))


        # 이미지를 비율 기준으로 축소
        # self._cvt_text_rec = cv2.resize(self._cvt_text_rec, dsize=(0, 0), fx=0.3, fy=0.3, interpolation=cv2.INTER_AREA)

        # 텍스트에 bounding box 생성하여 이미지의 텍스트 영역만 cut
        self._cvt_text_rec = self.bounding_box(self._cvt_text_rec)

        return self._cvt_text_rec

    def bounding_box(self, img):
        mser = cv2.MSER_create()
        regions, _ = mser.detectRegions(img)

        hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]

        start_x = 1e998
        start_y = 1e998
        end_x = 0
        end_y = 0
        for j, cnt in enumerate(hulls):
            x, y, w, h = cv2.boundingRect(cnt)
            if (x < start_x):
                start_x = x
            if (x + w > end_x):
                end_x = x + w
            if (y < start_y):
                start_y = y
            if (y + h > end_y):
                end_y = y + h
        if hulls != []:
            cropped_img = img[start_y:end_y, start_x:end_x]
        else:
            cropped_img = img
        
        # cv2.rectangle(img, (start_x, start_y), (end_x, end_y), (0, 255, 0), 1)
        return cropped_img

    # kernel과 일치하는 부분에 하나라도 0이 있으면 해당 부분을 모두 채움
    def dilation(self, img, kernel_size):
        # kernel size가 커질수록 영향 커짐
        kernel = np.ones(kernel_size, np.uint8)
        result = cv2.dilate(img, kernel, iterations=1)
        return result

    # kernel과 일치하는 부분에 하나라도 0이 있으면 해당 부분을 모두 제거
    def erosion(self, img, kernel_size):
        # kernel size가 커질수록 영향 커짐
        kernel = np.ones(kernel_size, np.uint8)
        result = cv2.erode(img, kernel, iterations=1)
        return result

    #  erosion 적용 후 dilation 적용, 잡티 제거 효과
    def opening(self, img, kernel_size):
        kernel = np.ones(kernel_size, np.uint8)
        result = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        return result

    # dilation 적용 후 erosion 적용, 윤곽이 도드라지는 효과
    def closing(self, img, kernel_size):
        kernel = np.ones(kernel_size, np.uint8)
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
