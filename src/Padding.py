import numpy as np


def padding_horizontal(image, current_size, desired_size):
    horizontal_padding_size = desired_size - current_size
    left_padding_size = horizontal_padding_size // 2
    right_padding_size = horizontal_padding_size - left_padding_size
    return np.pad(image, ((0,0), (left_padding_size, right_padding_size)), mode='constant')

def padding_vertical(image, current_size, desired_size):
    vertical_padding_size = desired_size - current_size
    top_padding_size = vertical_padding_size // 2
    bottom_padding_size = vertical_padding_size - top_padding_size
    return np.pad(image, ((top_padding_size, bottom_padding_size), (0,0)), mode='constant')
    
def cut_horizontal(image, current_size, desired_size):
    horizontal_cut_size = current_size - desired_size
    left_cut_start = horizontal_cut_size // 2
    return image[:, left_cut_start:left_cut_start + desired_size]

def cut_vertical(image, current_size, desired_size):
    vertical_cut_size = current_size - desired_size
    top_cut_start = vertical_cut_size // 2
    return image[top_cut_start:top_cut_start + desired_size, :]