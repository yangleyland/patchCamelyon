import tensorflow as tf
import h5py
import os
from openslide import OpenSlide
from PIL import Image
# import numpy as np
import tensorflow.experimental.numpy as tnp
import glob
import random
from tensorflow.keras.applications.resnet50 import preprocess_input,ResNet50
import numpy as np

tnp.experimental_enable_numpy_behavior()

# Define the size of the patches you want to extract
patch_size = (256, 256)  # Change this to your required patch size

# Directory with the .h5 files
h5_dir = '/home/emilysk8/TCGA_KICH_patches/patches/'

# Directory with the .svs files
svs_dir = '/home/emilysk8/TCGA_KICH/'

model_path = './'

classifier_model = tf.saved_model.load(model_path)
infer = classifier_model.signatures["serving_default"]
print(infer.structured_outputs)
# classifier_model = tf.keras.models.load_model(model_path)


def preprocess_patch(patch):
    patch = patch.resize((96, 96))  # Resize patch
    patch_np = np.array(patch)  # Convert PIL Image to NumPy array
    patch_np = np.repeat(patch_np[:, :, np.newaxis], 3, axis=2)  # Convert grayscale to 3 channels
    return patch_np

resnet_model = ResNet50(weights='imagenet', include_top=False, input_shape=(96, 96, 3))
count=0
for h5_path in glob.glob(h5_dir + '*.h5'):
    # Open the H5 file
    count+=1
    try:
        with h5py.File(h5_path, 'r') as h5:
            coords = h5['coords'][:]
    except OSError as e:
        print(f"Error opening file {h5_path}: {e}")
        continue  # Skip to the next file

    # Open the corresponding SVS file
    svs_path = os.path.join(svs_dir, os.path.splitext(os.path.basename(h5_path))[0] + '.svs')
    if os.path.exists(svs_path):
        slide = OpenSlide(svs_path)
    else:
        print(svs_path)
        continue
    slide_predictions = []
    # Randomly sample 3 coordinates (or however many you need)
    sampled_coords = random.sample(list(coords), 3)
    for coord in sampled_coords:
        patch = slide.read_region((coord[0], coord[1]), 0, patch_size).convert('L')  # Convert to grayscale
        patch_np = preprocess_patch(patch)
        patch_np = preprocess_input(patch_np)
        features = resnet_model.predict(np.expand_dims(patch_np, axis=0))
        # Use the classifier model for inference
        infer = classifier_model.signatures["serving_default"]

        # Assuming 'features' is your input tensor
        prediction = infer(tf.constant(features, dtype=tf.float32))['dense_1']
        slide_predictions.append(prediction)
        print(prediction)
    aggregate_prediction = np.mean(slide_predictions, axis=0)

    print(f"Aggregate prediction for slide {os.path.basename(svs_path)}: {aggregate_prediction}")
    threshold = 0.5
    if aggregate_prediction >= threshold:
        classification = "malignant"
    else:
        classification = "benign"
    
    print(f"Slide {os.path.basename(svs_path)} is classified as {classification}")
print(count)