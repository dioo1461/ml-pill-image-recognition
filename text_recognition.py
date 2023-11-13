from PIL import Image
import cv2
import pytesseract
import sys
import os
from image_preprocessing import Image_Preprocessing
sys.path.append(os.pardir)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'

# 현재 스크립트의 절대 경로
script_dir = os.path.dirname(os.path.abspath(__file__))

# 이미지 파일의 상대 경로
img_path = os.path.join(script_dir, 'exam_image.jpg')

converted = Image_Preprocessing(img_path)
img = converted.get_cvt_gray()
# img = cv2.imread(img_path)
text = pytesseract.image_to_string(img, lang='eng')
if text == '':
    print('text not detected')
else:
    print(text)

cv2.imshow('converted', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
