from PIL import Image
import cv2
import pytesseract
import sys
import os
from image_preprocessing import Image_Preprocessing
import image_processor
# from cvt_text_bounding import Cvt_Text_Bounding
import numpy as np
from CRAFT.test import Craft_Model

sys.path.append(os.pardir)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'

# 현재 스크립트의 절대 경로
script_dir = os.path.dirname(os.path.abspath(__file__))

def crop_image_by_coord(image, coord):
    start_x, start_y, end_x, end_y = coord
    cropped_image = image[int(start_y):int(end_y), int(start_x):int(end_x)]
    return cropped_image

def get_whole_text_area(image, all_coordinates):
    max_x = 0.0
    max_y = 0.0
    min_x = 9999999999.9
    min_y = 9999999999.9
    for i in all_coordinates:
        x_values = i[:, 0]
        y_values = i[:, 1]
        _min_x, _max_x = np.min(x_values), np.max(x_values)
        _min_y, _max_y = np.min(y_values), np.max(y_values)
        min_x, max_x = min(_min_x, min_x), max(_max_x, max_x)
        min_y, max_y = min(_min_y, min_y), max(_max_y, max_y)

    # 이미지를 잘라냅니다.
    cropped_image = image[int(min_y):int(max_y), int(min_x):int(max_x)]
    return cropped_image

def get_character_area(coordinates):
    # 좌표 배열에서 최소 및 최대 x 및 y 값을 찾습니다.
    x_values = coordinates[:, 0]
    y_values = coordinates[:, 1]
    min_x, max_x = np.min(x_values), np.max(x_values)
    min_y, max_y = np.min(y_values), np.max(y_values)
    return [min_x, min_y, max_x, max_y]

def get_box_precedence(box, cols):
    tolerance_factor = 10
    return ((box[1] // tolerance_factor) * tolerance_factor) * cols + box[0]

def get_sorted_bb(boxes, image):
    sorted_boxes = sorted(boxes, key=lambda x: get_box_precedence(x, image.shape[1]))
    return sorted_boxes

def get_boundings(img, craft):
    # cv2.imshow(f'origin', img)
    res = craft.run(img)

    # # get whole text area
    # cropped = get_whole_text_area(img, res)
    # processor = Image_Preprocessing(cropped)
    # preprocessed = processor.get_engraved_text_img_cvt()
    # preprocessed = processor.adaptive_threshold(preprocessed, 9, 2)
    # # _, preprocessed = cv2.threshold(cropped, 17, 255, cv2.THRESH_BINARY)
    # cv2.imshow(f'res', cv2.resize(preprocessed, (256, 256)))

    # get each characters
    coords = []

    for i in res:
        # print(i)
        cropped = get_character_area(i)
        coords.append(cropped)
        # processor = Image_Preprocessing(cropped)
        # preprocessed = processor.get_engraved_text_img_cvt()
        # preprocessed = processor.adaptive_threshold(cropped, 11, 3)
        # _, preprocessed = cv2.threshold(cropped, 17, 255, cv2.THRESH_BINARY)
        # cv2.imshow(f'{cnt} res {img_path}', cv2.resize(preprocessed, (256, 256)))
    sorted = get_sorted_bb(coords, img)

    cnt = 0
    res = []
    for i in sorted:
        bounded_img = crop_image_by_coord(img, i)
        processor = Image_Preprocessing(bounded_img)
        preprocessed = processor.get_engraved_text_img_cvt()
        # bounded_img = cv2.equalizeHist(bounded_img)
        preprocessed = processor.adaptive_threshold(preprocessed, 9, -1.5)
        preprocessed = processor.closing(preprocessed, (5,5))
        # cv2.imshow(f'{cnt} bounding', preprocessed)
        res.append(preprocessed)
        cnt += 1
    return res

###############
## main code ##
###############

# # 텍스트 바운딩 가중치 설정
# weightfile = './CRAFT/model/craft_mlt_25k.pth'
# text_threshold = 0.7  # 텍스트 상태 임계치
# low_text = 0.4  # 1에 가까울수록 bounding 영역이 작아짐
# link_threshold = 1  # 1에 가까울수록 word보다 character 기준 검출
# cuda = False

# # 모델 객체 생성
# craft = Craft_Model(weightfile, text_threshold, low_text, link_threshold, cuda)

# # 이미지 파일의 상대 경로
# img_path_1 = os.path.join(script_dir, 'data/' + 'Js.jpg')
# img_path_2 = os.path.join(script_dir, 'data/' + 'am5_example.jpg')
# img_path_3 = os.path.join(script_dir, 'data/' + 'js_example.jpg')
# img_path_4 = os.path.join(script_dir, 'data/' + 'exam_image.jpg')
# img_path_5 = os.path.join(script_dir, 'data/' + '7dot5.jpg')
# img_path_6 = os.path.join(script_dir, 'data/' + 'etx.jpg')
# img_path_7 = os.path.join(script_dir, 'data/' + 'B+C.jpg')

# get_boundings(img_path_1, craft)
# get_boundings(img_path_2, craft)
# get_boundings(img_path_3, craft)
# get_boundings(img_path_4, craft)
# get_boundings(img_path_5, craft)
# get_boundings(img_path_6, craft)
# get_boundings(cv2.imread(img_path_7), craft)

# img = cv2.imread(img_path_2)
# convertor = Image_Preprocessing(img)
# conv = convertor.get_engraved_text_img_cvt()
# # _, conv = cv2.threshold(conv, 35, 255, cv2.THRESH_BINARY)
# cv2.imshow('converted', conv)

# cvt1 = Cvt_Text_Bounding(cv2.imread(img_path_1))
# cvt1.get_text_boundings()

# text = pytesseract.image_to_string(img, lang='eng')
# if text == '':
#     print('text not detected')
# else:
#     print(text)


cv2.waitKey(0)
cv2.destroyAllWindows()
