from numpy import mean
import numpy as np
import tensorflow as tf

def dice_score(z_true, z_pred, mask=None):
    z_true_flat = tf.cast(tf.reshape(z_true, [-1]), tf.float32)
    z_pred_flat = tf.cast(tf.reshape(z_pred, [-1]), tf.float32)
    if mask is not None:
        mask_flat = tf.cast(tf.reshape(mask, [-1]), tf.float32)
        z_pred_flat = mask_flat * z_pred_flat
    intersection = tf.reduce_sum(z_true_flat * z_pred_flat)
    return (2.0 * intersection) / (tf.reduce_sum(z_true_flat) + tf.reduce_sum(z_pred_flat))

def dice_score_group(z_true, z_pred, mask):
    scores = []
    for i in range(z_true.shape[0]):
        score = dice_score(z_true[i], z_pred[i], mask[i])
        scores.append(score)
    print("Group scores:", scores)
    return np.mean(scores)
'''
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
'''