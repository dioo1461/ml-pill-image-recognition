from pilldata import PillData
import numpy as np

base_dir = 'D:\\data_of_ml\\dataset'
pill_data = PillData(base_dir)
test_d, test_l, training_d, training_l = pill_data.load_data()

test_d_example = next(iter(test_d.take(1)))
test_l_example = test_l[0]
training_d_example = next(iter(training_d.take(1)))
training_l_example = training_l[0]

# Print example data
print("Test Data Example:", test_d_example)
print("Test Label Example:", test_l_example)
print("Training Data Example:", training_d_example)
print("Training Label Example:", training_l_example)

# Print the shape of the training data and labels
print("Shape of Training Data:", pill_data.shape_training_d())
print("Shape of Training Labels:", pill_data.shape_training_l())
