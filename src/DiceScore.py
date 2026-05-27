from numpy import mean
import numpy as np
import tensorflow as tf


def dice_score_mask(images1, images2, masks):
    scores = []
    images2 = tf.squeeze(images2, axis=-1)  # Elimina el canal extra
    print("images 1 shape:", images1.shape)
    print("images 2 shape:", images2.shape)
    for i in range(images1.shape[0]):
        image1_list = image_to_list(images1[i])
        image2_list = image_to_list(images2[i])
        image_size = image1_list.shape[0]
        intersection_size = 0
        mask_list = image_to_list(masks[i])
        mask_zeros = 0
        for j in range(image_size):
            if mask_list[j] == 1:
                if image1_list[j].numpy() == image2_list[j].numpy():
                    intersection_size += 1
            else:
                mask_zeros += 1
        scores.append(intersection_size/(image_size - mask_zeros))
    return mean(scores)

def image_to_list(image):
    return tf.reshape(image, [-1])