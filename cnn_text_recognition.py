import tensorflow as tf
import keras
from keras import layers
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import numpy as np
import io
from image_preprocessing import Image_Preprocessing
import fetch_db


NUM_PRINT_FRONT_VARIATIONS = 1
NUM_PRINT_BACK_VARIATIONS = 400

LEARNING_RATE = 0.001
EPOCH = 50

(train_data_x, train_data_y), (test_data_x, test_data_y) = fetch_db.get_print_data(1, 10, True)

# train_data_y, test_data_y에서 고유한 레이블을 추출
unique_labels = list(set(train_data_y + test_data_y))

# 문자열 레이블을 정수로 매핑하는 사전 생성
label_mapping = {label: idx for idx, label in enumerate(unique_labels)}

# 매핑된 레이블로 데이터 변환
train_data_y_numeric = [label_mapping[label] for label in train_data_y]
test_data_y_numeric = [label_mapping[label] for label in test_data_y]

for i in range(len(test_data_x)):
    img = Image.open(io.BytesIO(test_data_x[i]))
    image_array = np.array(img)
    convertor = Image_Preprocessing(image_array)
    converted = convertor.get_engraved_text_img_cvt(False)
    test_data_x[i] = converted

for i in range(len(train_data_x)):
    img = Image.open(io.BytesIO(train_data_x[i]))
    image_array = np.array(img)
    convertor = Image_Preprocessing(image_array)
    converted = convertor.get_engraved_text_img_cvt(False)
    train_data_x[i] = converted

cv2.destroyAllWindows()

# 모델 구축
model = keras.Sequential()

# Input Layer
model.add(layers.InputLayer((1080, 1080, 1)))

# Feature Extractor(Backbone) : Conv + Pooling
# Conv로 특징 추출 -> Pooling으로 사이즈 줄이기 -> filter수로 channel 증가시키기
model.add(layers.Conv2D(filters=16,  # 필터 개수
                        kernel_size=(3, 3),  # 필터 사이즈 (h, w)
                        # 패딩 방식 설정(valid:패딩안함 / same:input, output사이즈변동없도록패딩추가)
                        padding="valid",
                        strides=(1, 1),  # 필터 이동 간격 설정 (h, w)
                        activation="relu"  # 활성함수 설정
                        ))
model.add(layers.MaxPool2D(pool_size=(2, 2),  # Max값 추출할 영역 크기 설정 (h, w)
                           strides=(2, 2),  # 이동 간격 (h, w)
                           # 패딩 방식 설정(valid:pool_size보다 작은 부분 버리기 / same:pool_size보다 작은 부분있으면 패딩 채워서 계산)
                           padding="valid",
                           ))
model.add(layers.Conv2D(filters=32, kernel_size=3,
                        padding="valid",
                        activation="relu"
                        ))
# model.add(layers.MaxPool2D(padding="valid"))
# model.add(layers.Conv2D(filters=64, kernel_size=3,
#           padding="same", activation="relu"))
model.add(layers.MaxPool2D(padding="valid"))

# Estimator : Dense(Fully Connected Layer)
model.add(layers.Flatten())
model.add(layers.Dense(units=256, activation="relu"))

# Output Layer
model.add(layers.Dense(units=NUM_PRINT_FRONT_VARIATIONS, activation="softmax"))


# optimizer 설정, 손실함수 설정, 평가지표 설정을 해준다.
model.compile(optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
              loss="categorical_crossentropy",
              metrics=["accuracy"])


# 학습 진행
hist = model.fit(x=np.array(train_data_x),
                  y=np.array(train_data_y_numeric),  # 위에서 batch size는 따로 지정해 줘서 생략가능
                  epochs=EPOCH,
                  # dataset을 사용할때는 validation_split말고 validation_data 사용한다.
                  validation_data=(np.array(test_data_x), np.array(test_data_y_numeric))
                  )

plt.subplot(1, 2, 1)
plt.plot(hist.epoch, hist.history["loss"], label="Train loss")
plt.plot(hist.epoch, hist.history["val_loss"], label="Validation loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(hist.epoch, hist.history["accuracy"], label="Train Accuracy")
plt.plot(hist.epoch, hist.history["val_accuracy"], label="Validation Accuracy")
plt.legend()

plt.tight_layout()
plt.show()

