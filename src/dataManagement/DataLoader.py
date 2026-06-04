from glob import glob
import numpy as np
import cv2
import os


ROOT_PATH = os.path.abspath("data")
TEST_PATH = os.path.join(ROOT_PATH, "test")
TRAIN_PATH = os.path.join(ROOT_PATH, "training")

TRAINING_IMAGES_PATH = os.path.join(TRAIN_PATH, "images")
TRAINING_MASKS_PATH = os.path.join(TRAIN_PATH, "mask")
TRAINING_MANUAL_PATH = os.path.join(TRAIN_PATH, "1st_manual")

TEST_IMAGES_PATH = os.path.join(TEST_PATH, "images")
TEST_MASKS_PATH = os.path.join(TEST_PATH, "mask")
TEST_1ST_MANUAL_PATH = os.path.join(TEST_PATH, "1st_manual")
TEST_2ND_MANUAL_PATH = os.path.join(TEST_PATH, "2nd_manual")


def load_image(path, is_binary):
    image = cv2.imread(path)
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

    images_paths = sorted(glob(os.path.join(TRAINING_IMAGES_PATH, "*")))
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

    images_paths = sorted(glob(os.path.join(TEST_IMAGES_PATH, "*")))
    masks_paths = sorted(glob(os.path.join(TEST_MASKS_PATH, "*")))
    manual1_paths = sorted(glob(os.path.join(TEST_1ST_MANUAL_PATH, "*")))
    manual2_paths = sorted(glob(os.path.join(TEST_2ND_MANUAL_PATH, "*")))
    for i in range(len(images_paths)): # Asumimos que hay una máscara por imagen
        images.append(load_image(images_paths[i], is_binary=False))
        masks.append(load_image(masks_paths[i], is_binary=True))
        manual1.append(load_image(manual1_paths[i], is_binary=True))
        manual2.append(load_image(manual2_paths[i], is_binary=True))
    return np.array(images), np.array(masks), np.array(manual1), np.array(manual2)