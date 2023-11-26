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

def crop_image(image, coordinates):
    # 좌표 배열에서 최소 및 최대 x 및 y 값을 찾습니다.
    x_values = coordinates[:, 0]
    y_values = coordinates[:, 1]
    min_x, max_x = np.min(x_values), np.max(x_values)
    min_y, max_y = np.min(y_values), np.max(y_values)

    # 이미지를 잘라냅니다.
    cropped_image = image[int(min_y):int(max_y), int(min_x):int(max_x)]

    return cropped_image

def convert_img(img_path):
    img = cv2.imread(img_path)
    convertor = Image_Preprocessing(img)
    processor = image_processor.ImageProcessor(img)
    processor.draw_binding_box()
    img_2 = processor.load_edged()

    converted = convertor.get_engraved_text_img_cvt(show_flag=True)
    print(img.shape)

    # cv2.imshow(f'converted {img_path}', converted)
    # cv2.imshow(f'resized {img_path}', cv2.resize(converted, (256, 256)))
    # cv2.imshow(f'boundings {img_path}', convertor.get_engraved_text_bounding())
    # cv2.imshow(f'resized {img_path}', cv2.resize(convertor.get_engraved_text_bounding(), (256, 256)))
    # cv2.imshow(f'img_2 {img_path}', img_2)
    # cv2.imshow(f'img_2 {img_path}', cv2.resize(img_2, (256, 256)))

    return img


# 텍스트 바운딩 가중치 설정
weightfile = './CRAFT/model/craft_mlt_25k.pth'
text_threshold = 0.7  # 텍스트 상태 임계치
low_text = 0.45  # 1에 가까울수록 bounding 영역이 작아짐
link_threshold = 1  # 1에 가까울수록 word보다 character 기준 검출
cuda = False

# 모델 객체 생성
craft = Craft_Model(weightfile, text_threshold, low_text, link_threshold, cuda)

def get_boundings(img_path):
    img = cv2.imread(img_path)
    pre = Image_Preprocessing(img)
    img = pre.get_engraved_text_img_cvt()
    cv2.imshow(f'origin {img_path}', img)
    res = craft.run(img)

    cnt = 0
    for i in res:
        print(i)
        cropped = crop_image(img, i)
        processor = Image_Preprocessing(cropped)
        # preprocessed = processor.get_engraved_text_img_cvt()
        # preprocessed = processor.adaptive_threshold(cropped, 11, 3)
        _, preprocessed = cv2.threshold(cropped, 17, 255, cv2.THRESH_BINARY)
        cv2.imshow(f'{cnt} res {img_path}', cv2.resize(preprocessed, (256, 256)))

        cnt+=1
# 이미지 파일의 상대 경로
img_path_1 = os.path.join(script_dir, 'data/' + 'Js.jpg')
img_path_2 = os.path.join(script_dir, 'data/' + 'M200.jpg')
img_path_3 = os.path.join(script_dir, 'data/' + 'exam_image.jpg')

# get_boundings(img_path_1)
get_boundings(img_path_2)
# get_boundings(img_path_3)

# cvt1 = Cvt_Text_Bounding(cv2.imread(img_path_1))
# cvt1.get_text_boundings()

# text = pytesseract.image_to_string(img, lang='eng')
# if text == '':
#     print('text not detected')
# else:
#     print(text)


cv2.waitKey(0)
cv2.destroyAllWindows()
