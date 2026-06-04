import numpy as np
import os
from dotenv import load_dotenv
from dataManagement.Padding import padding_horizontal, padding_vertical, cut_horizontal, cut_vertical


load_dotenv()

HORIZONTAL_UNET_SIZE = int(os.getenv("HORIZONTAL_UNET_SIZE"))
VERTICAL_UNET_SIZE = int(os.getenv("VERTICAL_UNET_SIZE"))


def transform_image(image):
    horizontal_size = image.shape[1]
    vertical_size = image.shape[0]
    transformed_image = image
    if horizontal_size < HORIZONTAL_UNET_SIZE:
        transformed_image = padding_horizontal(transformed_image, current_size=horizontal_size, desired_size=HORIZONTAL_UNET_SIZE)
    elif horizontal_size > HORIZONTAL_UNET_SIZE:
        transformed_image = cut_horizontal(transformed_image, current_size=horizontal_size, desired_size=HORIZONTAL_UNET_SIZE)
    if vertical_size < VERTICAL_UNET_SIZE:
        transformed_image = padding_vertical(transformed_image, current_size=vertical_size, desired_size=VERTICAL_UNET_SIZE)
    elif vertical_size > VERTICAL_UNET_SIZE:
        transformed_image = cut_vertical(transformed_image, current_size=vertical_size, desired_size=VERTICAL_UNET_SIZE)
    return transformed_image
    
def transform(images, masks=None):
    transformed = []
    for i in range(len(images)):
        image = images[i]
        if masks is not None:
            image = masks[i] * image
        transformed.append(transform_image(image))
    return np.array(transformed)