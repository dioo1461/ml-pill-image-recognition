import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
import os
from rembg import remove
from PIL import Image
import pilldata as pdt


class Convert_Image():
    def __init__(self, img_path):
        self._image = cv2.imread(img_path)

    def get_cvt_gray(self):
        image_rbg = remove(self._image)
        image_rbg_gray = cv2.cvtColor(image_rbg, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(image_rbg_gray, (5, 5), sigmaX=0)
        enhanced_image = cv2.convertScaleAbs(blur, alpha=2.8, beta=20)
        self._cvt_gray = enhanced_image
        # _, binary_image = cv2.threshold(blur, 150, 255, cv2.THRESH_BINARY)  # 이진화
        return self._cvt_gray

# edged = cv2.Canny(blur, 10, 250)
# ret, thresh = cv2.threshold(edged, 127, 255, cv2.THRESH_BINARY)
# contours, hierarchy = cv2.findContours(
#     edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


# cnt = contours[0]
# rect = cv2.minAreaRect(cnt)
# box = cv2.boxPoints(rect)
# box = np.intp(box)
# image_bb = cv2.drawContours(edged, [box], 0, (255, 255, 255), 2)
# # cv2.imshow('image', image1)
# # cv2.imshow('remove background', image_rbg)
# # cv2.imshow('blur', blur)
# # cv2.imshow('Edged', edged)
# cv2.imshow('bounding box', image_bb)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
