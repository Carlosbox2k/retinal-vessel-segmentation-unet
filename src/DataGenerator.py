import tensorflow as tf
from glob import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib
from keras.models import Model
from keras.optimizers import Adam
import os


ROOT_PATH = os.path.abspath("data")
TEST_PATH = os.path.join(ROOT_PATH, "test")
TRAIN_PATH = os.path.join(ROOT_PATH, "training")
HORIZONTAL_SIZE = int(os.getenv("HORIZONTAL_SIZE", 565))
VERTICAL_SIZE = int(os.getenv("VERTICAL_SIZE", 584))
DATA_AUGMENTATION_NUMBER = int(os.getenv("DATA_AUGMENTATION_NUMBER", 40))

training_images_path = os.path.join(TRAIN_PATH, "images")
training_masks_path = os.path.join(TRAIN_PATH, "mask")

print(f"Training images path: {training_images_path}")

def load_image(path, is_mask):
    print(f"Loading {"mask" if is_mask else "image"} from: {path}")
    image = cv2.imread(path)
    image = cv2.resize(image, (HORIZONTAL_SIZE, VERTICAL_SIZE))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)   # shape: (size,size,3) -> (size,size,1)
    if is_mask:
        image = image/255   # normalize (int)
    else:
        image = image/255.  # normalize (float)
    return image

def load_data(images_path, masks_path):
    images = []
    masks = []

    images_paths = sorted(glob(images_path + "\\*"))
    masks_paths = sorted(glob(masks_path + "\\*"))
    for i in range(len(images_paths)): # Asumimos que hay una máscara por imagen
        images.append(load_image(images_paths[i], is_mask=False))
        masks.append(load_image(masks_paths[i], is_mask=True))
    return np.array(images), np.array(masks)

def data_augmentation(image, mask):
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)  # Flip horizontal
        mask = cv2.flip(mask, 1)
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 0)  # Flip vertical
        mask = cv2.flip(mask, 0)
    if np.random.rand() < 0.5:
        angle = np.random.uniform(-15, 15)
        M = cv2.getRotationMatrix2D((HORIZONTAL_SIZE/2, VERTICAL_SIZE/2), angle, 1)
        image = cv2.warpAffine(image, M, (HORIZONTAL_SIZE, VERTICAL_SIZE))
        mask = cv2.warpAffine(mask, M, (HORIZONTAL_SIZE, VERTICAL_SIZE))
    if np.random.rand() < 0.5:
        scale = np.random.uniform(0.8, 0.9)
        M = cv2.getRotationMatrix2D((HORIZONTAL_SIZE/2, VERTICAL_SIZE/2), 0, scale)
        image = cv2.warpAffine(image, M, (HORIZONTAL_SIZE, VERTICAL_SIZE))
        mask = cv2.warpAffine(mask, M, (HORIZONTAL_SIZE, VERTICAL_SIZE))
    
    mask = np.round(mask) # Para que la máscara siga siendo binaria después de la transformación en los bordes
    return image, mask

def data_augmentation2(image, mask):
    scale = np.random.uniform(0.8, 0.9)
    M = cv2.getRotationMatrix2D((HORIZONTAL_SIZE/2, VERTICAL_SIZE/2), 0, scale)
    image = cv2.warpAffine(image, M, (HORIZONTAL_SIZE, VERTICAL_SIZE))
    mask = cv2.warpAffine(mask, M, (HORIZONTAL_SIZE, VERTICAL_SIZE))

    mask = np.round(mask)
    return image, mask

def augment_data(X, y):
    augmented_images = []
    augmented_masks = []
    for i in range(DATA_AUGMENTATION_NUMBER):
        n = np.random.randint(0, len(X))
        im, mk = data_augmentation(X[n], y[n])
        augmented_images.append(im)
        augmented_masks.append(mk)
    return np.array(augmented_images), np.array(augmented_masks)

def append_augmented_data(X, y):
    augmented_X, augmented_y = augment_data(X, y)
    for i in range(DATA_AUGMENTATION_NUMBER):
        n = np.random.randint(0, len(X)+1)
        X = np.insert(X, n, augmented_X[i], axis=0)
        y = np.insert(y, n, augmented_y[i], axis=0)
    return X, y

X, y = load_data(training_images_path, training_masks_path)
X, y = append_augmented_data(X, y)
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

im, mk = data_augmentation2(X[0], y[0])
fig, ax = plt.subplots(2,2, figsize=(10,5))
print(f"X shape: {im.shape}")
print(f"y shape: {mk.shape}")

print(f"X shape: {X[0].shape}")
print(f"y shape: {y[0].shape}")
ax[0,0].imshow(im, cmap='gray')
ax[0,0].set_title('Image')
ax[0,1].imshow(mk, cmap='gray')
ax[0,1].set_title('Mask')
ax[1,0].imshow(X[0], cmap='gray')
ax[1,0].set_title('Image')
ax[1,1].imshow(y[0], cmap='gray')
ax[1,1].set_title('Mask')
plt.show()


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