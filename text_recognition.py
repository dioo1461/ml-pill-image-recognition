from PIL import Image
import cv2
import pytesseract
import sys
import os
from image_preprocessing import Image_Preprocessing
import image_processor

sys.path.append(os.pardir)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'

# 현재 스크립트의 절대 경로
script_dir = os.path.dirname(os.path.abspath(__file__))

def convert_img(img_path):
    img = cv2.imread(img_path)
    convertor = Image_Preprocessing(img)
    processor = image_processor.ImageProcessor(img)
    processor.draw_binding_box()
    img_2 = processor.load_edged()

    converted = convertor.get_engraved_text_rec_cvt()
    print(img.shape)

    cv2.imshow(f'converted {img_path}', converted)
    cv2.imshow(f'resized {img_path}', cv2.resize(converted, (256, 256)))
    cv2.imshow(f'boundings {img_path}', convertor.get_engraved_text_bounding())
    cv2.imshow(f'resized {img_path}', cv2.resize(convertor.get_engraved_text_bounding(), (256, 256)))
    cv2.imshow(f'img_2 {img_path}', img_2)
    cv2.imshow(f'img_2 {img_path}', cv2.resize(img_2, (256, 256)))

    return img

# 이미지 파일의 상대 경로
img_path_1 = os.path.join(script_dir, 'data/' + 'js_example.jpg')
img_path_3 = os.path.join(script_dir, 'data/' + 'Js.jpg')
img_path_2 = os.path.join(script_dir, 'data/' + 'M200.jpg')

img_1 = convert_img(img_path_1)
img_2 = convert_img(img_path_2)
img_3 = convert_img(img_path_3)


# text = pytesseract.image_to_string(img, lang='eng')
# if text == '':
#     print('text not detected')
# else:
#     print(text)

cv2.waitKey(0)
cv2.destroyAllWindows()
