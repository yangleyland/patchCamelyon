import tensorflow as tf
import h5py
import os
from openslide import OpenSlide
from PIL import Image
# import numpy as np
import tensorflow.experimental.numpy as tnp
import glob
import random
from tensorflow.keras.applications.resnet50 import preprocess_input

tnp.experimental_enable_numpy_behavior()

# Define the size of the patches you want to extract
patch_size = (256, 256)  # Change this to your required patch size

# Directory with the .h5 files
h5_dir = '/home/emilysk8/TCGA-KICH_patches/patches'

# Directory with the .svs files
svs_dir = '/home/emilysk8/TCGA-KICH/'

model_path = './'

model = tf.saved_model.load(model_path)

def preprocess_patch(patch):
    # Convert to grayscale, resize, and expand dimensions if necessary
    patch_resized = patch.resize((96, 96))  # Resize to the expected input size of the model
    patch_np = tnp.array(patch_resized)
    patch_np = tnp.expand_dims(patch_np, axis=-1)  # Add a channel dimension
    patch_np = preprocess_input(patch_np)  # Preprocess for ResNet50
    return patch_np

for h5_path in glob.glob(h5_dir + '*.h5'):
    # Open the H5 file
    with h5py.File(h5_path, 'r') as h5:
        # Load the 'coords' dataset
        coords = h5['coords'][:]

    # Open the corresponding SVS file
    svs_path = os.path.join(svs_dir, os.path.splitext(os.path.basename(h5_path))[0] + '.svs')
    slide = OpenSlide(svs_path)

    # Randomly sample 3 coordinates (or however many you need)
    sampled_coords = random.sample(list(coords), 3)
    for coord in sampled_coords:
        # Extract a patch from the SVS file
        patch = slide.read_region((coord[0], coord[1]), 0, patch_size).convert('L')  # Convert to grayscale

        # Preprocess the patch
        patch_processed = preprocess_patch(patch)

        prediction = model(tnp.array([patch_processed]))  # Ensure to wrap patch in a batch
        print(f"Prediction for patch at {coord}: {prediction}")