import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
import os
from rembg import remove
from PIL import Image
import pilldata as pd
import positionManager

class Image_Preprocessing():
    def __init__(self, image):
        self._image = image
        self._text_img_cvt = None
        self._engraved_text_img_cvt = None
        self._engraved_text_bounding = None

    def init_by_path(self, img_path):
        self._image = cv2.imread(img_path)

    def get_engraved_text_img_cvt(self):
        if self._engraved_text_img_cvt is not None:
            return self._engraved_text_img_cvt

        self._engraved_text_img_cvt = self._image

        # # 이미지 배경 제거
        # self._engraved_text_img_cvt = remove(self._image)

        # 이미지의 대비를 증가시킴
        # self._engraved_text_img_cvt = self.apply_clahe(self._engraved_text_img_cvt)

        # 이미지를 흑백으로 변환
        self._engraved_text_img_cvt = cv2.cvtColor(self._engraved_text_img_cvt, cv2.COLOR_BGR2GRAY)

        # 가우시안 블러 처리
        self._engraved_text_img_cvt = cv2.GaussianBlur(self._engraved_text_img_cvt, (5, 5), sigmaX=0)

        # histogram equalization 적용
        self._engraved_text_img_cvt = cv2.equalizeHist(self._engraved_text_img_cvt)
    
        # black hat 적용
        self._engraved_text_img_cvt = self.blackhat(self._engraved_text_img_cvt, (51,51))

        # 단순 threshold 이진화 적용
        # _, self._engraved_text_img_cvt = cv2.threshold(self._engraved_text_img_cvt, 35, 255, cv2.THRESH_BINARY)

        # adaptive threshold 이진화 적용
        # self._engraved_text_img_cvt = self.adaptive_threshold(self._engraved_text_img_cvt, 9, 5)

        # # 이진화된 이미지 반전 (외곽선 검출 시 어두운 배경에서 밝은 객체를 찾는 것이 더 유리)
        # self._engraved_text_img_cvt = cv2.bitwise_not(self._engraved_text_img_cvt)

        # # 외곽선 제거
        # contours, _ = cv2.findContours(self._engraved_text_img_cvt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # # 빈 이미지 생성
        # contour_image = np.zeros_like(self._engraved_text_img_cvt)
        # # 모든 윤곽선 그리기

        # # cv2.drawContours(self._engraved_text_img_cvt, [contours[0]], -1, (0, 0, 0), thickness=cv2.FILLED)

        # # self._engraved_text_img_cvt = self.opening(self._engraved_text_img_cvt, (5,5))

        # [s_x, e_x, s_y, e_y] = self.get_bounding_box_coordinates(self._engraved_text_img_cvt)
        # self._engraved_text_img_cvt = self._engraved_text_img_cvt[s_y:e_y, s_x:e_x]

        self._engraved_text_img_cvt = cv2.resize(self._engraved_text_img_cvt, (256, 256))

        return self._engraved_text_img_cvt

    def get_engraved_text_bounding(self, show_flag:bool):
        if self._engraved_text_bounding is not None:
            return self._engraved_text_bounding
        # 이미지 배경 제거
        res = remove(self._image)

        # 이미지의 대비를 증가시킴
        res = self.apply_clahe(res)

        # 이미지를 흑백으로 변환
        res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

        # 가우시안 블러 처리
        res = cv2.GaussianBlur(res, (5, 5), sigmaX=0)

        # histogram equalization 적용
        res = cv2.equalizeHist(res)

        # adaptive threshold 이진화 적용
        # block size 작을수록, C값 클수록 threshold 엄격해짐
        res = self.adaptive_threshold(res, 13, 3)

        # cv2.imshow('pre-res', res)

        # 외곽선 검출 후 제거 작업
        contours, last_idx = positionManager.GetPillContour(self._image, 50)
        res = cv2.drawContours(res, contours, last_idx, (255,255,255), thickness=10)

        # res = cv2.bitwise_not(res)

        # tmp = np.zeros_like(res)
        # cv2.imshow('contours', tmp)
        # tmp = cv2.drawContours(tmp, contours, last_idx, (0,0,0), thickness=10)

        res = self.closing(res, (5,5))

        # res = self.crop_bounding_box(res)

        self._engraved_text_bounding = res

        if show_flag:
            cv2.imshow(f'engraved_text_bounding', self._engraved_text_bounding)

        return self._engraved_text_bounding

    def get_bounding_box_coordinates(img_binary):
        contours, _ = cv2.findContours(img_binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        all_points = []

        for contour in contours:
            for point in contour:
                all_points.extend(point)

        if all_points:
            all_points = np.array(all_points).squeeze()
            rect = cv2.minAreaRect(all_points)
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            # 최소, 최대 좌표 계산
            min_x = np.min(box[:, 0])
            min_y = np.min(box[:, 1])
            max_x = np.max(box[:, 0])
            max_y = np.max(box[:, 1])

            return min_x, min_y, max_x, max_y

        # 외곽선이 없는 경우 None 반환
        return None

    def get_printed_text_img_cvt(self, show_flag:bool):
        if self._text_img_cvt is not None:
            return self._text_img_cvt
        # 이미지 배경 제거
        self._text_img_cvt = remove(self._image)

        # # 이미지의 대비를 증가시킴
        # self._text_img_cvt = self.apply_clahe(self._text_img_cvt)

        # 이미지를 흑백으로 변환
        self._text_img_cvt = cv2.cvtColor(self._text_img_cvt, cv2.COLOR_BGR2GRAY)

        # 가우시안 블러 처리
        self._text_img_cvt = cv2.GaussianBlur(self._text_img_cvt, (5, 5), sigmaX=0)

        # 밝기와 대비 증가
        self._text_img_cvt = cv2.convertScaleAbs(self._text_img_cvt, alpha=2.8, beta=20)

        # adaptive threshold 적용 (이진화)
        self._text_img_cvt = self.adaptive_threshold(self._text_img_cvt, 9, 5)

        # 이진화된 이미지 반전
        self._text_img_cvt = cv2.bitwise_not(self._text_img_cvt)

        # morphological 변환 적용
        self._text_img_cvt = self.closing(self._text_img_cvt, (5,5))
        self._text_img_cvt = self.opening(self._text_img_cvt, (5,5))
        self._text_img_cvt = self.dilation(self._text_img_cvt, (9,7))

        # # 가우시안 블러 처리
        # self._text_img_cvt = cv2.GaussianBlur(self._text_img_cvt, (5, 5), sigmaX=0)

        # 이미지를 비율 기준으로 축소
        # self._text_img_cvt = cv2.resize(self._text_img_cvt, dsize=(0, 0), fx=0.3, fy=0.3, interpolation=cv2.INTER_AREA)

        # 텍스트에 bounding box 생성하여 이미지의 텍스트 영역만 cut
        self._text_img_cvt = self.crop_bounding_box(self._text_img_cvt)

        if show_flag:
            cv2.imshow(f'printed_text_convert', self._text_img_cvt)

        return self._text_img_cvt

    # block size 작을수록, C값 클수록 threshold 엄격해짐
    def adaptive_threshold(self, img, block_size, C):
        result = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, C)
        return result
    

    def crop_bounding_box(self, img):
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
        
        cv2.rectangle(img, (start_x, start_y), (end_x, end_y), (0, 255, 0), 1)
        return cropped_img

    def blackhat(self, img, kernel_size):
        # kernel size가 커질수록 영향 커짐
        # kernel = np.ones(kernel_size, np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        result = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
        return result

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

    # 이미지의 대비를 증가시킴 (clahe 균일화 적용)
    def apply_clahe(self, img):
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