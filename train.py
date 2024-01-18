from tensorflow import keras
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D

# Load the dataset
custom_data_dir = "/home/yangleyland"
df, info = tfds.load('patch_camelyon', with_info=True, as_supervised=True, data_dir=custom_data_dir)

# Getting the train, validation and test data
train_data = df['train']
valid_data = df['validation']
test_data = df['test']

# Preprocessing function: Convert to grayscale
def preprocess(image, label):
    image = tf.image.rgb_to_grayscale(image)
    image = tf.image.resize(image, [96, 96])  # Resize if needed
    return image, label

# Apply preprocessing to the datasets
train_data = train_data.map(preprocess)
valid_data = valid_data.map(preprocess)
test_data = test_data.map(preprocess)

# Batch and prefetch the data for optimal performance
BATCH_SIZE = 32
train_data = train_data.batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)
valid_data = valid_data.batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)
test_data = test_data.batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)

# Initialize ResNet50 for feature extraction
resnet_model = ResNet50(weights='imagenet', include_top=False, input_shape=(96, 96, 3))

# Feature extraction function
def extract_features(data_batch):
    images, labels = next(iter(data_batch.unbatch().batch(1000)))  # Adjust the batch size according to your memory capacity
    images = tf.repeat(images, 3, axis=-1)  # Convert grayscale to 3 channels
    features = resnet_model.predict(preprocess_input(images), verbose=1)
    return features, labels

# Extract features (This might take a while and consume significant memory)
train_features, train_labels = extract_features(train_data)
valid_features, valid_labels = extract_features(valid_data)
test_features, test_labels = extract_features(test_data)

# Build and Train the Classifier
classifier_model = Sequential([
    # Flattening the output of ResNet50
    Flatten(input_shape=(3, 3, 2048)),

    # Dense layer
    Dense(64, activation='relu'),

    # Dropout layer to reduce overfitting
    Dropout(0.5),

    # Output layer
    Dense(1, activation='sigmoid')
])


classifier_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
classifier_model.fit(train_features, train_labels, validation_data=(valid_features, valid_labels), epochs=10)

# save model path
model_save_path = "/home/yangleyland"  # or '.savedmodel' for SavedModel format
classifier_model.save(model_save_path)