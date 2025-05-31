import os
import numpy as np
import tensorflow as tf
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt

def load_images_and_labels(base_dir, image_size=(256, 256)):
    data = []
    labels = []
    
    # 이미지 이름의 첫 글자를 라벨로 사용하는 데 필요한 라벨 매핑
    label_mapping = {}
    
    for subdir, dirs, files in os.walk(base_dir):
        for file in files:
            file_path = os.path.join(subdir, file)
            image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if image is not None and image.shape == image_size:
                # 이미지 이름의 첫 글자 추출
                label_char = file[0]
                label_char = label_char.upper()
                # 라벨 매핑 업데이트
                if label_char not in label_mapping:
                    label_mapping[label_char] = len(label_mapping)
                label = label_mapping[label_char]
                
                data.append(image.reshape(image_size + (1,)))
                labels.append(label)

    data = np.array(data, dtype='float32') / 255.0
    labels = np.array(labels)

    # 데이터셋 무작위로 섞기
    indices = np.arange(len(data))
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]

    return data, labels, label_mapping

def train_model(data, labels, batch_size=4, epochs=200):
    num_classes = len(np.unique(labels))

    # 모델 구조 정의
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(256, 256, 1)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.6),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)  # 학습률 설정
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # 모델 학습
    history = model.fit(data, labels, batch_size=batch_size, epochs=epochs, validation_split=0.2)

    return model, history

def main():
    base_dir = 'inputs\\final'

    data, labels, label_mapping = load_images_and_labels(base_dir)
    print("Label Mapping:", label_mapping)
    model, history = train_model(data, labels)

    # 훈련 과정 시각화
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.show()

    response = input("Would you like to save the model? (yes/no): ")
    if response.lower() == 'yes':
      model.save('text_rec_model.h5')
      print("Model saved as text_rec_model.h5")
    else:
      print("Model not saved.")
if __name__ == '__main__':
    main()