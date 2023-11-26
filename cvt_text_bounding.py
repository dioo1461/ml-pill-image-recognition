import subprocess
import os
import cv2

class Cvt_Text_Bounding():
    def __init__(self, image):
        self._image = image
        self._boundings = None
        self._bounding_coords = None

    def get_text_boundings(self):
        if self._boundings is not None:
            return self._boundings
        
        input_folder = './temp/'    
        if not os.path.isdir(input_folder):
            os.mkdir(input_folder)

        input_file_name = "input.jpg"
        # 이미지 저장
        cv2.imwrite(input_file_name, self._image)

        # 필요한 인자들 설정
        weightfile = './CRAFT-pytorch/model/craft_mlt_25k.pth'
        test_folder = './temp'
        text_threshold = '0.5'  # 텍스트 상태 임계치
        low_text = '0.4'  # 1에 가까울수록 bounding 영역이 작아짐
        link_threshold = '1'  # 1에 가까울수록 word보다 character 기준 검출
        cuda = 'false'

        # 명령어 생성
        command = f'''python ./CRAFT-pytorch/test.py --trained_model={weightfile} --test_folder={test_folder} --text_threshold={text_threshold} --low_text={low_text} --link_threshold={link_threshold} --cuda={cuda}'''

        # 서브프로세스 실행
        subp = subprocess.Popen(command)
        subp.communicate()

        output_path = './result/res_input.txt'
        self._bounding_coords = get_input_ops(open(output_path, "r"))
        print(self._bounding_coords)

def get_input_ops(file):
    op_list = []
    while True:
        line = file.readline()
        if not line:
            break
        line = line.strip()
        column_values = line.split()
        op_list.append(column_values)
    return op_list
        
cvt = Cvt_Text_Bounding(None)
print(cvt.get_text_boundings())