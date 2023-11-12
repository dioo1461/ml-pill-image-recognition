import numpy as np
import os
import json
import tensorflow as tf



class PillData:

  def __init__(self, base_dir):
    self.base_dir = base_dir

  def load_data(self):
    # Load test and training data and labels
    test_d = self._load_images(os.path.join(self.base_dir, 'test_set', 'test_data'))
    test_l = self._load_labels(os.path.join(self.base_dir, 'test_set', 'test_label'))
    training_d = self._load_images(os.path.join(self.base_dir, 'training_set', 'training_data'))
    training_l = self._load_labels(os.path.join(self.base_dir, 'training_set', 'training_label'))

    return test_d, test_l, training_d, training_l

  def _load_images(self, dir_path):
    # Load images from subdirectories
    file_paths = []
    for root, dirs, files in os.walk(dir_path):
      for file in files:
        if file.endswith('.png'):
          file_paths.append(os.path.join(root, file))

    list_ds = tf.data.Dataset.from_tensor_slices(file_paths)
    images_ds = list_ds.map(self._process_image)
    return images_ds

  def _process_image(self, file_path):
    # Use tf.py_function to handle tensor to string conversion
    img = tf.py_function(self._read_and_process_image, [file_path], tf.float32)
    return img

  def _read_and_process_image(self, file_path):
    # file_path is now a regular Python object (not a tf.Tensor)
    file_path = file_path.numpy().decode('utf-8')

    # Read and process image as before
    img = tf.io.read_file(file_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, [256, 256])
    img = img / 255.0
    return img

  def _load_labels(self, dir_path):
    # Load labels from subdirectories
    labels = []
    for root, dirs, files in os.walk(dir_path):
      for file in files:
        if file.endswith('.json'):
          file_path = os.path.join(root, file)
          with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            labels.append(data)
    return labels

  def shape_training_d(self):
    # Return the shape of training data
    training_d = self._load_images(os.path.join(self.base_dir, 'training_set', 'training_data'))
    return tf.data.experimental.cardinality(training_d).numpy()

  def shape_training_l(self):
    # Return the shape of training labels
    training_l = self._load_labels(os.path.join(self.base_dir, 'training_set', 'training_label'))
    return len(training_l)