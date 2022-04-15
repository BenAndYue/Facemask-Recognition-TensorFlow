# import libraries

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import cv2

image_size = (128,128)

base_dir = '../data/testing/'


train_gen = keras.preprocessing.image.ImageDataGenerator(
    horizontal_flip=True,
    rescale=1./255
)

val_gen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

test_gen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_ds = train_gen.flow_from_directory(base_dir+'Train',target_size=image_size,seed=42)
val_ds = val_gen.flow_from_directory(base_dir+'Validation',target_size=image_size,seed=42)
test_ds = test_gen.flow_from_directory(base_dir+'Test',target_size=image_size,seed=42)