import cv2 as cv
import numpy as np

def GetPillContour(img_color, highThreshold = 50, showFlag = False):
    img_gray = cv.cvtColor(img_color, cv.COLOR_BGR2GRAY)
    img_gray = cv.medianBlur(img_gray, 5)
    if showFlag:
        cv.imshow("Blur", img_gray)
        cv.waitKey(0)

    img_colorTmp = img_color.copy()

    low = 0
    img_canny = cv.Canny(img_gray, low, highThreshold)
    if showFlag:
        cv.imshow("img_canny", img_canny)
        cv.waitKey(0)

    adaptive = cv.adaptiveThreshold(img_canny, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 15, 0)

    # contour에서 구멍 난 경우를 대비
    kernel = np.ones((10, 10), np.uint8)
    closing = cv.morphologyEx(adaptive, cv.MORPH_CLOSE, kernel)
    contourInput = closing
    contours, hierarchy = cv.findContours(contourInput, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    maxIdx = 0
    idx = 0
    maxArea = 0.0

    for cnt in contours:
        area = cv.contourArea(cnt)
        if maxArea < area:
            maxIdx = idx
            maxArea = area
        idx = idx + 1

    cv.drawContours(img_colorTmp, contours, idx - 1, (0, 0, 255), 1)
    if showFlag:
        cv.imshow("img_colorTmp", img_colorTmp)
        cv.waitKey(0)

        # 알약 외곽선을 추출한다.
    #contour = contours[idx - 1]
    lastIdx = idx - 1
    return contours, maxIdx#lastIdx
