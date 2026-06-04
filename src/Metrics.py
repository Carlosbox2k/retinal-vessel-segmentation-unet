from numpy import mean
import tensorflow as tf
import keras


@keras.saving.register_keras_serializable() # Registra la función en el diccionario de objetos personalizados de Keras
def dice_score_loss(z_true, z_pred):
    z_true_flat = tf.cast(tf.reshape(z_true, [-1]), tf.float32)
    z_pred_flat = tf.cast(tf.reshape(z_pred, [-1]), tf.float32)
    intersection = tf.reduce_sum(z_true_flat * z_pred_flat)
    return 1 - (2.0 * intersection) / (tf.reduce_sum(z_true_flat) + tf.reduce_sum(z_pred_flat))

@keras.saving.register_keras_serializable()
def bce_dice_loss(z_true, z_pred):
    bce = tf.keras.losses.binary_crossentropy(z_true, z_pred)
    dice = dice_score_loss(z_true, z_pred)
    return bce + dice

@keras.saving.register_keras_serializable()
def dice_score(z_true, z_pred, mask=None):
    z_true_flat = tf.cast(tf.reshape(z_true, [-1]), tf.float32)
    z_pred_flat = tf.cast(tf.reshape(z_pred, [-1]), tf.float32)
    z_pred_flat = tf.round(z_pred_flat)  # Binarizar las predicciones
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
    return mean(scores)