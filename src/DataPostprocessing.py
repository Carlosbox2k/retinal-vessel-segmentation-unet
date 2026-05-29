import numpy as np
import os
from dotenv import load_dotenv
from Padding import cut_horizontal, cut_vertical, padding_horizontal, padding_vertical


load_dotenv()

HORIZONTAL_UNET_SIZE = int(os.getenv("HORIZONTAL_UNET_SIZE", 576))
VERTICAL_UNET_SIZE = int(os.getenv("VERTICAL_UNET_SIZE", 592))

def detransform_image(image, original_image_size):
    horizontal_size = original_image_size[1]
    vertical_size = original_image_size[0]
    detransformed_image = image
    if horizontal_size < HORIZONTAL_UNET_SIZE:
        detransformed_image = cut_horizontal(detransformed_image, current_size=HORIZONTAL_UNET_SIZE, desired_size=horizontal_size)
    elif horizontal_size > HORIZONTAL_UNET_SIZE:
        detransformed_image = padding_horizontal(detransformed_image, current_size=HORIZONTAL_UNET_SIZE, desired_size=horizontal_size)
    if vertical_size < VERTICAL_UNET_SIZE:
        detransformed_image = cut_vertical(detransformed_image, current_size=VERTICAL_UNET_SIZE, desired_size=vertical_size)
    elif vertical_size > VERTICAL_UNET_SIZE:
        detransformed_image = padding_vertical(detransformed_image, current_size=VERTICAL_UNET_SIZE, desired_size=vertical_size)
    return detransformed_image

def detransform(images, images_sizes):
    transformed = []
    for i in range(images.shape[0]):
        transformed.append(detransform_image(images[i], images_sizes[i]))
    return np.array(transformed)