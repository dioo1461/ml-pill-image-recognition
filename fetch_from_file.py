from PIL import Image
import cv2
import pytesseract
import sys
import os
from image_preprocessing import Image_Preprocessing
import image_processor
import numpy as np
from CRAFT.test import Craft_Model
import text_recognition
from tqdm import tqdm

# 이미지를 읽어와서 전처리 후 저장하는 함수
def process_images_in_folder(folder_path, root_folder, idx, craft:Craft_Model):
    # 폴더 이름 가져오기
    folder_name = os.path.basename(folder_path)

    # 폴더 내의 이미지 파일들의 경로 리스트 가져오기
    image_paths = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # 전처리 결과물을 저장할 폴더 생성
    output_folder = f'.\\{root_folder}\\{folder_name}\\{folder_name} result'
    os.makedirs(output_folder, exist_ok=True)

    # 이미지를 읽어와서 전처리 후 저장
    for index_img, image_path in enumerate(image_paths):
        # 이미지 읽기
        original_image = cv2.imread(image_path)

        # 전처리
        boundings = text_recognition.get_boundings(original_image, craft)
        
        splited = list(folder_name)
        for index_bounding, bounding in enumerate(boundings):
            # 저장할 파일 경로 생성
            if index_bounding >= len(splited):  
                output_filename = f"exception {index_bounding}"
            else:
                output_filename = f'{splited[index_bounding]} {idx} {index_img} {index_bounding}.jpg'
            output_path = os.path.join(output_folder, output_filename)
            # output_path = os.path.join(output_path, output_filename)
            # cv2.imshow(f'{index_bounding}', bounding)
            # 전처리 결과물 저장

            try:
                cv2.imwrite(output_path, bounding)
                print(f'Saved: {output_path}')
            except cv2.error as e:
                print(f"Error occured while processing image '{folder_name} idx {index_img}': {e}")
                continue  # 오류가 발생하면 다음 이미지로 계속 진행

############
### MAIN ###
############

# 텍스트 바운딩 가중치 설정
weightfile = './CRAFT/model/craft_mlt_25k.pth'
text_threshold = 0.7  # 텍스트 상태 임계치
low_text = 0.4  # 1에 가까울수록 bounding 영역이 작아짐
link_threshold = 1  # 1에 가까울수록 word보다 character 기준 검출
cuda = False

# 모델 객체 생성
craft = Craft_Model(weightfile, text_threshold, low_text, link_threshold, cuda)

root_folder = 'inputs\\sanghyeok'

# 루트 폴더 내의 모든 하위 폴더 순회
for idx, folder_name in enumerate(tqdm(os.listdir(root_folder), desc=f"conveting images")):
    if idx >= 4 or idx <= 2:
        continue
    print(f"converting {folder_name}")
    folder_path = os.path.join(root_folder, folder_name)
    # 하위 폴더인 경우에만 처리
    if os.path.isdir(folder_path):
        process_images_in_folder(folder_path, root_folder, idx, craft)







