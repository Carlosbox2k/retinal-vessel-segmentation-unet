from glob import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib
from keras.models import Model
from keras.optimizers import Adam
import os
from dotenv import load_dotenv

from DataPreprocesing import padding, unpadding

load_dotenv()

ROOT_PATH = os.path.abspath("data")
TEST_PATH = os.path.join(ROOT_PATH, "test")
TRAIN_PATH = os.path.join(ROOT_PATH, "training")
HORIZONTAL_SIZE = int(os.getenv("HORIZONTAL_SIZE"))
VERTICAL_SIZE = int(os.getenv("VERTICAL_SIZE"))
DATA_AUGMENTATION_NUMBER = int(os.getenv("DATA_AUGMENTATION_NUMBER", 40))

HORIZONTAL_PADDING_SIZE = int(os.getenv("HORIZONTAL_PADDING_SIZE", 576))
VERTICAL_PADDING_SIZE = int(os.getenv("VERTICAL_PADDING_SIZE", 592))

training_images_path = os.path.join(TRAIN_PATH, "images")
training_masks_path = os.path.join(TRAIN_PATH, "mask")
training_manual_path = os.path.join(TRAIN_PATH, "1st_manual")

print(f"Training images path: {training_images_path}")

def load_image(path, is_binary):
    image = cv2.imread(path)
    image = cv2.resize(image, (HORIZONTAL_SIZE, VERTICAL_SIZE))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)   # shape: (size,size,3) -> (size,size,1)
    if is_binary:
        image = image/255   # normalize (int)
    else:
        image = image/255.  # normalize (float)
    return image

def load_data(images_path, masks_path, manual_path):
    images = []
    masks = []
    manual = []

    images_paths = sorted(glob(images_path + "\\*"))
    masks_paths = sorted(glob(masks_path + "\\*"))
    manual_paths = sorted(glob(manual_path + "\\*"))
    for i in range(len(images_paths)): # Asumimos que hay una máscara por imagen
        images.append(load_image(images_paths[i], is_binary=False))
        masks.append(load_image(masks_paths[i], is_binary=True))
        manual.append(load_image(manual_paths[i], is_binary=True))
    return np.array(images), np.array(masks), np.array(manual)

def data_augmentation(image, mask, manual):
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)  # Flip horizontal
        mask = cv2.flip(mask, 1)
        manual = cv2.flip(manual, 1)
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 0)  # Flip vertical
        mask = cv2.flip(mask, 0)
        manual = cv2.flip(manual, 0)
    if np.random.rand() < 0.5:
        angle = np.random.uniform(-15, 15)
        M = cv2.getRotationMatrix2D((HORIZONTAL_SIZE/2, VERTICAL_SIZE/2), angle, 1)
        image = cv2.warpAffine(image, M, (HORIZONTAL_SIZE, VERTICAL_SIZE))
        mask = cv2.warpAffine(mask, M, (HORIZONTAL_SIZE, VERTICAL_SIZE))
        manual = cv2.warpAffine(manual, M, (HORIZONTAL_SIZE, VERTICAL_SIZE))
    if np.random.rand() < 0.5:
        scale = np.random.uniform(0.8, 0.9)
        M = cv2.getRotationMatrix2D((HORIZONTAL_SIZE/2, VERTICAL_SIZE/2), 0, scale)
        image = cv2.warpAffine(image, M, (HORIZONTAL_SIZE, VERTICAL_SIZE))
        mask = cv2.warpAffine(mask, M, (HORIZONTAL_SIZE, VERTICAL_SIZE))
        manual = cv2.warpAffine(manual, M, (HORIZONTAL_SIZE, VERTICAL_SIZE))
    
    mask = np.round(mask) # Para que la máscara siga siendo binaria después de la transformación en los bordes
    manual = np.round(manual)
    return image, mask, manual

def data_augmentation2(image, mask, manual):
    scale = np.random.uniform(0.8, 0.9)
    M = cv2.getRotationMatrix2D((HORIZONTAL_SIZE/2, VERTICAL_SIZE/2), 0, scale)
    image = cv2.warpAffine(image, M, (HORIZONTAL_SIZE, VERTICAL_SIZE))
    mask = cv2.warpAffine(mask, M, (HORIZONTAL_SIZE, VERTICAL_SIZE))
    manual = cv2.warpAffine(manual, M, (HORIZONTAL_SIZE, VERTICAL_SIZE))
    
    mask = np.round(mask)
    manual = np.round(manual)
    return image, mask, manual

def augment_data(X, y, z):
    augmented_images = []
    augmented_masks = []
    augmented_manual = []
    for i in range(DATA_AUGMENTATION_NUMBER):
        n = np.random.randint(0, len(X))
        im, mk, mn = data_augmentation(X[n], y[n], z[n])
        augmented_images.append(im)
        augmented_masks.append(mk)
        augmented_manual.append(mn)
    return np.array(augmented_images), np.array(augmented_masks), np.array(augmented_manual)

def append_augmented_data(X, y, z):
    augmented_X, augmented_y, augmented_z = augment_data(X, y, z)
    for i in range(DATA_AUGMENTATION_NUMBER):
        n = np.random.randint(0, len(X)+1)
        X = np.insert(X, n, augmented_X[i], axis=0)
        y = np.insert(y, n, augmented_y[i], axis=0)
        z = np.insert(z, n, augmented_z[i], axis=0)
    return X, y, z

X, y, z = load_data(training_images_path, training_masks_path, training_manual_path)
X, y, z = append_augmented_data(X, y, z)
'''
print("X shape:", X.shape)
print("y shape:", y.shape)

n = len(X)
fig, ax = plt.subplots(1,2, figsize=(10,5))
ax[0].imshow(X[n-1], cmap='gray')
ax[0].set_title('Image')
ax[1].imshow(y[n-1], cmap='gray')
ax[1].set_title('Mask')
plt.show()
'''

fig, ax = plt.subplots(1,3, figsize=(10,5))
im = X[0]
im_padded = padding(im)
im_unpadded = unpadding(im_padded)
print(f"X shape: x: {im.shape[1]} y: {im.shape[0]}")
print(f"X padded shape: x: {im_padded.shape[1]} y: {im_padded.shape[0]}")
print(f"X unpadded shape: x: {im_unpadded.shape[1]} y: {im_unpadded.shape[0]}")
ax[0].imshow(im, cmap='gray')
ax[0].set_title('Image')
ax[1].imshow(im_padded, cmap='gray')
ax[1].set_title('Image padded')
ax[2].imshow(im_unpadded, cmap='gray')
ax[2].set_title('Image unpadded')
plt.show()
'''
im, mk, mn = data_augmentation2(X[0], y[0], z[0])
fig, ax = plt.subplots(2,3, figsize=(10,5))
print(f"X shape: {im.shape}")
print(f"y shape: {mk.shape}")
print(f"z shape: {mn.shape}")
print(f"X shape: {X[0].shape}")
print(f"y shape: {y[0].shape}")
print(f"z shape: {z[0].shape}")
ax[0,0].imshow(im, cmap='gray')
ax[0,0].set_title('Image')
ax[0,1].imshow(mk, cmap='gray')
ax[0,1].set_title('Mask')
ax[0,2].imshow(mn, cmap='gray')
ax[0,2].set_title('Manual')
ax[1,0].imshow(X[0], cmap='gray')
ax[1,0].set_title('Image')
ax[1,1].imshow(y[0], cmap='gray')
ax[1,1].set_title('Mask')
ax[1,2].imshow(z[0], cmap='gray')
ax[1,2].set_title('Manual')
plt.show()
'''

'''
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
fig, ax = plt.subplots(1,2, figsize=(10,5))
ax[0].imshow(X[0], cmap='gray')
ax[0].set_title('Image')
ax[1].imshow(y[0], cmap='gray')
ax[1].set_title('Mask')
plt.show()
'''