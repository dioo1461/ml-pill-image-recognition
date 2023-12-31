import pymysql
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from image_preprocessing import Image_Preprocessing
from dotenv import load_dotenv
import os
import text_recognition
from CRAFT.test import Craft_Model
import fetch_db

def load_encoding_table():
    load_dotenv()
    host = os.environ.get('Host')
    port = os.environ.get('Port')
    user = os.environ.get('User')
    password = os.environ.get('Password')
    db = os.environ.get('DB')

    # mySql과의 커넥션 정의
    connection = pymysql.connect(host=host, port=int(
        port), user=user, password=password, db=db, cursorclass=pymysql.cursors.DictCursor)
    encoding_table = {}
    try:
        with connection.cursor() as cursor:
            sql = "SELECT `print_front`, `print_back` FROM `pill_data`"
            cursor.execute(sql)
            results = cursor.fetchall()

            unique_texts = set()
            for row in results:
                front_text = row['print_front'] if row['print_front'] is not None else ''
                back_text = row['print_back'] if row['print_back'] is not None else ''
                front_text = front_text.replace(' '|'분할선', '')
                back_text = back_text.replace(' '|'분할선', '')
                unique_texts.add(front_text)
                unique_texts.add(back_text)
            
            encoding_table = {text: i for i, text in enumerate(unique_texts)}
    finally:
        connection.close()
    
    return encoding_table
    # x_train, y_train, x_test, y_test = fetch_db.get_print_data(30, 10, True)
    # return x_train, y_train, x_test, y_test

def load_data(craft_model:Craft_Model, limit=1000):
    load_dotenv()
    host = os.environ.get('Host')
    port = os.environ.get('Port')
    user = os.environ.get('User')
    password = os.environ.get('Password')
    db = os.environ.get('DB')

    # mySql과의 커넥션 정의
    connection = pymysql.connect(host=host, port=int(
        port), user=user, password=password, db=db, cursorclass=pymysql.cursors.DictCursor)
    images = []
    labels = []
    try:
        with connection.cursor() as cursor:
            sql = f"""
                SELECT i.image, l.print_front, l.print_back, l.drug_dir
                FROM image_data i
                JOIN label_data l ON i.file_name = l.file_name 
                order by rand() 
                LIMIT {limit}
            """
            cursor.execute(sql, (limit,))
            data = cursor.fetchall()

            for item in tqdm(data, desc="Loading and preprocessing images"):
                nparr = np.frombuffer(item['image'], np.uint8)
                # image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                image = cv2.imread(image)

                # Image_Preprocessing 클래스를 사용하여 이미지 처리
                img_processor = Image_Preprocessing(image)

                # 수정된 부분: preprocess_image 메서드를 호출
                craft_model.run(image)
                processed_img = img_processor.get_engraved_text_img_cvt()
                preprocessed = img_processor.adaptive_threshold(preprocessed, 9, 0)
                # cv2.imshow("1", processed_img)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows
                images.append(processed_img)

                label_text = item['print_front'] if item['drug_dir'] == '앞면' else item['print_back']
                label = encoding_table.get(label_text, -1)
                labels.append(label)
    finally:
        connection.close()
    
    print(images, labels)
    return np.array(images), np.array(labels)



# 텍스트 바운딩 가중치 설정
weightfile = './CRAFT/model/craft_mlt_25k.pth'
text_threshold = 0.7  # 텍스트 상태 임계치
low_text = 0.4  # 1에 가까울수록 bounding 영역이 작아짐
link_threshold = 1  # 1에 가까울수록 word보다 character 기준 검출
cuda = False

# 모델 객체 생성
craft = Craft_Model(weightfile, text_threshold, low_text, link_threshold, cuda)

encoding_table = load_encoding_table()
images, labels = load_data(craft, 50)

# 데이터 분할
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)
val_images, test_images, val_labels, test_labels = train_test_split(test_images, test_labels, test_size=0.5, random_state=42)

train_images = train_images / 255.0
test_images = test_images / 255.0
val_images = val_images / 255.0

# 모델 구성
model = tf.keras.models.Sequential()
# 첫 번째 컨볼루션 레이어
model.add(tf.keras.layers.Conv2D(8, (3, 3), activation='relu', input_shape=(256, 256, 1)))
model.add(tf.keras.layers.MaxPooling2D((3, 3)))
model.add(tf.keras.layers.Dropout(0.25))

# 두 번째 컨볼루션 레이어
model.add(tf.keras.layers.Conv2D(8, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((3, 3)))
model.add(tf.keras.layers.Dropout(0.25))

# 세 번째 컨볼루션 레이어
model.add(tf.keras.layers.Conv2D(16, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Dropout(0.25))

# 전역 평균 풀링 레이어
model.add(tf.keras.layers.GlobalAveragePooling2D())

# Dense 레이어
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(10, activation='softmax')) 


# 모델 컴파일
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=200, validation_data=(val_images, val_labels))

# 결과 그래프 출력
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()

# 테스트 세트에서 모델 평가
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\nTest accuracy:', test_acc)


model.save('text_rec.h5')
print("Model saved as 'text_rec.h5'")
