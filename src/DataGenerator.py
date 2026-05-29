from glob import glob
import numpy as np
import cv2
import os
from dotenv import load_dotenv


load_dotenv()

HORIZONTAL_UNET_SIZE = int(os.getenv("HORIZONTAL_UNET_SIZE", 576))
VERTICAL_UNET_SIZE = int(os.getenv("VERTICAL_UNET_SIZE", 592))
DATA_AUGMENTATION_NUMBER = int(os.getenv("DATA_AUGMENTATION_NUMBER", 40))

ROOT_PATH = os.path.abspath("data")
TEST_PATH = os.path.join(ROOT_PATH, "test")
TRAIN_PATH = os.path.join(ROOT_PATH, "training")

TRAINING_IMAGE_PATH = os.path.join(TRAIN_PATH, "images")
TRAINING_MASKS_PATH = os.path.join(TRAIN_PATH, "mask")
TRAINING_MANUAL_PATH = os.path.join(TRAIN_PATH, "manual_1st")

TEST_IMAGE_PATH = os.path.join(TEST_PATH, "images")
TEST_MASKS_PATH = os.path.join(TEST_PATH, "mask")
TEST_1ST_MANUAL_PATH = os.path.join(TEST_PATH, "1st_manual")
TEST_2ND_MANUAL_PATH = os.path.join(TEST_PATH, "2nd_manual")

def load_image(path, is_binary):
    image = cv2.imread(path)
    #image = cv2.resize(image, (HORIZONTAL_UNET_SIZE, VERTICAL_UNET_SIZE))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)   # shape: (size,size,3) -> (size,size,1)
    if is_binary:
        image = image/255   # normalize (int)
    else:
        image = image/255.  # normalize (float)
    return image

def load_training_data():
    images = []
    masks = []
    manual = []

    images_paths = sorted(glob(os.path.join(TRAINING_IMAGE_PATH, "*")))
    masks_paths = sorted(glob(os.path.join(TRAINING_MASKS_PATH, "*")))
    manual_paths = sorted(glob(os.path.join(TRAINING_MANUAL_PATH, "*")))
    for i in range(len(images_paths)): # Asumimos que hay una máscara por imagen
        images.append(load_image(images_paths[i], is_binary=False))
        masks.append(load_image(masks_paths[i], is_binary=True))
        manual.append(load_image(manual_paths[i], is_binary=True))
    return np.array(images), np.array(masks), np.array(manual)

def load_data_test():
    images = []
    masks = []
    manual1 = []
    manual2 = []

    images_paths = sorted(glob(os.path.join(TEST_IMAGE_PATH, "*")))
    masks_paths = sorted(glob(os.path.join(TEST_MASKS_PATH, "*")))
    manual1_paths = sorted(glob(os.path.join(TEST_1ST_MANUAL_PATH, "*")))
    manual2_paths = sorted(glob(os.path.join(TEST_2ND_MANUAL_PATH, "*")))
    for i in range(len(images_paths)): # Asumimos que hay una máscara por imagen
        images.append(load_image(images_paths[i], is_binary=False))
        masks.append(load_image(masks_paths[i], is_binary=True))
        manual1.append(load_image(manual1_paths[i], is_binary=True))
        manual2.append(load_image(manual2_paths[i], is_binary=True))
    return np.array(images), np.array(masks), np.array(manual1), np.array(manual2)

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