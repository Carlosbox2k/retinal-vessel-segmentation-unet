import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()

HORIZONTAL_SIZE = int(os.getenv("HORIZONTAL_SIZE", 565))
VERTICAL_SIZE = int(os.getenv("VERTICAL_SIZE", 584))
HORIZONTAL_PADDING_SIZE = int(os.getenv("HORIZONTAL_PADDING_SIZE", 576))
VERTICAL_PADDING_SIZE = int(os.getenv("VERTICAL_PADDING_SIZE", 592))

def padding(image):
    pad_height = max(0, VERTICAL_PADDING_SIZE - VERTICAL_SIZE)
    pad_width = max(0, HORIZONTAL_PADDING_SIZE - HORIZONTAL_SIZE)
    
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left
    
    padded_image = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant')
    
    return padded_image

def unpadding(image):
    height, width = image.shape[:2]
    pad_height = max(0, height - VERTICAL_SIZE)
    pad_width = max(0, width - HORIZONTAL_SIZE)
    
    pad_top = pad_height // 2
    pad_left = pad_width // 2
    
    unpadded_image = image[pad_top:pad_top+VERTICAL_SIZE, pad_left:pad_left+HORIZONTAL_SIZE]
    
    return unpadded_image

def mask_and_padding(X, y):
    padded_X = []
    for i in range(len(X)):
        masked = y[i] * X[i]
        padded_X.append(padding(masked))
    return np.array(padded_X)