import numpy as np
import cv2
import os
from dotenv import load_dotenv


load_dotenv()

DATA_AUGMENTATION_SIZE = int(os.getenv("DATA_AUGMENTATION_SIZE"))


def data_augmentation(image, mask, manual):
    height, width = image.shape[:2]
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
        M = cv2.getRotationMatrix2D((width/2, height/2), angle, 1)
        image = cv2.warpAffine(image, M, (width, height))
        mask = cv2.warpAffine(mask, M, (width, height))
        manual = cv2.warpAffine(manual, M, (width, height))
    if np.random.rand() < 0.5:
        scale = np.random.uniform(0.8, 0.9)
        M = cv2.getRotationMatrix2D((width/2, height/2), 0, scale)
        image = cv2.warpAffine(image, M, (width, height))
        mask = cv2.warpAffine(mask, M, (width, height))
        manual = cv2.warpAffine(manual, M, (width, height))
    
    mask = np.round(mask)   # Para que la máscara siga siendo binaria después de la transformación en los bordes
    manual = np.round(manual)
    return image, mask, manual

def augment_data(x, y, z):
    augmented_images = []
    augmented_masks = []
    augmented_manual = []
    for i in range(DATA_AUGMENTATION_SIZE):
        n = np.random.randint(0, len(x))
        im, mk, mn = data_augmentation(x[n], y[n], z[n])
        augmented_images.append(im)
        augmented_masks.append(mk)
        augmented_manual.append(mn)
    return np.array(augmented_images), np.array(augmented_masks), np.array(augmented_manual)

def append_augmented_data(x, y, z):
    augmented_x, augmented_y, augmented_z = augment_data(x, y, z)
    for i in range(DATA_AUGMENTATION_SIZE):
        n = np.random.randint(0, len(x)+1)
        x = np.insert(x, n, augmented_x[i], axis=0)
        y = np.insert(y, n, augmented_y[i], axis=0)
        z = np.insert(z, n, augmented_z[i], axis=0)
    return x, y, z