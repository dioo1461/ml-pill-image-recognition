import tensorflow as tf
from keras import layers, models

# 데이터셋 로드
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# 데이터 전처리
train_images, test_images = train_images / 255.0, test_images / 255.0

# 모델 구축
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),     # 이미지를 1D로 펼치기
    layers.Dense(128, activation='relu'),      # Fully Connected Layer
    layers.Dropout(0.2),
    layers.Dense(10)                            # 출력 레이어 (숫자 0-9)
])

# 모델 컴파일
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# 모델 훈련
model.fit(train_images, train_labels, epochs=5)

# 모델 평가
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\n테스트 정확도:', test_acc)

# 새로운 데이터에 대한 예측
predictions = model.predict(new_data)