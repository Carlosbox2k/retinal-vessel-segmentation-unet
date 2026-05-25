import tensorflow as tf
from glob import glob
import numpy as np
from sklearn.model_selection import train_test_split
import cv2
import matplotlib.pyplot as plt
import matplotlib
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate
from keras.optimizers import Adam
#from tensorflow.keras.metrics import *
import os


ROOT_PATH = os.path.abspath("data")
TEST_PATH = os.path.join(ROOT_PATH, "test")
TRAIN_PATH = os.path.join(ROOT_PATH, "training")

training_images_path = os.path.join(TRAIN_PATH, "images")
training_masks_path = os.path.join(TRAIN_PATH, "mask")
print(f"Training images path: {training_images_path}")
def load_image(path):
    print(f"Loading image from: {path}")
    # Usar np.fromfile y cv2.imdecode porque cv2.imread falla con caracteres especiales en la ruta (como "3º")
    image = cv2.imread(path)
    
    # image = cv2.resize(image, (size,size))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)   # shape: (size,size,3) -> (size,size,1)
    image = image/255.   # normalize
    return image

def load_data(images_path, masks_path):
    images = []
    masks = []

    images_paths = sorted(glob(images_path + "\\*"))
    masks_paths = sorted(glob(masks_path + "\\*"))

    for i in range(len(images_paths)): # Asumimos que hay una máscara por imagen
        images.append(load_image(images_paths[i]))
        masks.append(load_image(masks_paths[i]))
    
    return np.array(images), np.array(masks)


X, y = load_data(training_images_path, training_masks_path)
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
fig, ax = plt.subplots(1,2, figsize=(10,5))
ax[0].imshow(X[0], cmap='gray')
ax[0].set_title('Image')
ax[1].imshow(y[0], cmap='gray')
ax[1].set_title('Mask')
fig.suptitle('Normal class', fontsize=16)
plt.show()