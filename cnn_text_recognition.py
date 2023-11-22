import tensorflow as tf
import keras
from keras import layers
import matplotlib.pyplot as plt
import pymysql
import cv2
from PIL import Image
import numpy as np
import io
from dotenv import load_dotenv
import os

NUM_PRINT_FRONT = 800
NUM_PRINT_BACK = 400

LEARNING_RATE = 0.001

# .env에서 커넥션 정보 불러오기
load_dotenv()
host = os.environ.get('Host')
port = os.environ.get('Port')
user = os.environ.get('User')
password = os.environ.get('Password')
db = os.environ.get('DB')

# mySql과의 커넥션 정의
# conn = pymysql.connect(host=host, port=int(port), user=user, password=password, db=db, charset='utf8mb4')

# 커넥션의 커서를 정의
cursor = conn.cursor()
cursor.execute("select image from image_data where file_name = 'K-032657_0_0_0_0_75_000_200.png'")
img = cursor.fetchone()

img = Image.open(io.BytesIO(img[0]))
image_array = np.array(img)
cv2.imshow('fetched', image_array)
cv2.waitKey(0)
cv2.destroyAllWindows()

# TODO 데이터셋 로드하는 코드 작성
# (train_x, train_y) = load_train_data()
# (test_x, test_y) = load_test_data()

# # 데이터 전처리
# train_images, test_images = train_images / 255.0, test_images / 255.0

# # 모델 구축
# model = keras.Sequential()

# # Input Layer
# model.add(layers.InputLayer((28, 28, 1)))

# # Feature Extractor(Backbone) : Conv + Pooling
# # Conv로 특징 추출 -> Pooling으로 사이즈 줄이기 -> filter수로 channel 증가시키기
# model.add(layers.Conv2D(filters=16,  # 필터 개수
#                         kernel_size=(3, 3),  # 필터 사이즈 (h, w)
#                         # 패딩 방식 설정(valid:패딩안함 / same:input, output사이즈변동없도록패딩추가)
#                         padding="valid",
#                         stride=(1, 1),  # 필터 이동 간격 설정 (h, w)
#                         activation="relu"  # 활성함수 설정
#                         ))
# model.add(layers.MaxPool2D(pool_size=(2, 2),  # Max값 추출할 영역 크기 설정 (h, w)
#                            strides=(2, 2),  # 이동 간격 (h, w)
#                            # 패딩 방식 설정(valid:pool_size보다 작은 부분 버리기 / same:pool_size보다 작은 부분있으면 패딩 채워서 계산)
#                            padding="valid",
#                            ))
# model.add(layers.Conv2D(filters=32, kernel_size=3,
#                         padding="valid",
#                         activation="relu"
#                         ))
# # model.add(layers.MaxPool2D(padding="valid"))
# # model.add(layers.Conv2D(filters=64, kernel_size=3,
# #           padding="same", activation="relu"))
# model.add(layers.MaxPool2D(padding="valid"))

# # Estimator : Dense(Fully Connected Layer)
# model.add(layers.Flatten())
# model.add(layers.Dense(units=256, activation="relu"))

# # Output Layer
# model.add(layers.Dense(units=NUM_PRINT_FRONT, activation="softmax"))


# # optimizer 설정, 손실함수 설정, 평가지표 설정을 해준다.
# model.compile(optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
#               loss="categorical_crossentropy",
#               metrics=["accuracy"])


# # 학습 진행
# hist = model.fit(x=train_x,
#                   y=train_y,  # 위에서 batch size는 따로 지정해 줘서 생략가능
#                   epochs=50,
#                   # dataset을 사용할때는 validation_split말고 validation_data 사용한다.
#                   validation_data=(test_x, test_y)
#                   )

# plt.subplot(1, 2, 1)
# plt.plot(hist.epoch, hist.history["loss"], label="Train loss")
# plt.plot(hist.epoch, hist.history["val_loss"], label="Validation loss")
# plt.legend()

# plt.subplot(1, 2, 2)
# plt.plot(hist.epoch, hist.history["accuracy"], label="Train Accuracy")
# plt.plot(hist.epoch, hist.history["val_accuracy"], label="Validation Accuracy")
# plt.legend()

# plt.tight_layout()
# plt.show()
