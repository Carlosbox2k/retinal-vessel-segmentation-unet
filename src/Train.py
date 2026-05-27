from keras.models import Model
import os
import tensorflow as tf
from UnetModel import build_model
from dotenv import load_dotenv
from sklearn.model_selection import KFold, cross_validate
from DataGenerator import load_data
from sklearn.metrics import make_scorer
import numpy as np
from DataPreprocesing import mask_and_padding

load_dotenv()

ROOT_PATH = os.path.abspath("data")
TEST_PATH = os.path.join(ROOT_PATH, "test")
TRAIN_PATH = os.path.join(ROOT_PATH, "training")
HORIZONTAL_PADDING_SIZE = int(os.getenv("HORIZONTAL_PADDING_SIZE", 576))
VERTICAL_PADDING_SIZE = int(os.getenv("VERTICAL_PADDING_SIZE", 592))

training_images_path = os.path.join(TRAIN_PATH, "images")
training_masks_path = os.path.join(TRAIN_PATH, "mask")
training_manual_path = os.path.join(TRAIN_PATH, "1st_manual")

def dice_score(z_true, z_pred, mask=None):
    z_true_flat = tf.reshape(z_true, [-1])
    z_pred_flat = tf.reshape(z_pred, [-1])
    if mask is not None:
        mask_flat = tf.reshape(mask, [-1])
        z_pred_flat = mask_flat * z_pred_flat
    intersection = tf.reduce_sum(z_true_flat * z_pred_flat)
    return (2.0 * intersection) / (tf.reduce_sum(z_true_flat) + tf.reduce_sum(z_pred_flat))

def dice_score_group (z_true, z_pred, mask):
    scores = []
    for i in range(z_true.shape[0]):
        score = dice_score(z_true[i], z_pred[i], mask[i])
        scores.append(score)
    return np.mean(scores)

MODEL = build_model(input_shape=(VERTICAL_PADDING_SIZE, HORIZONTAL_PADDING_SIZE, 1))
MODEL.compile(loss="binary_crossentropy", optimizer="Adam", metrics=[dice_score])

tf.keras.utils.plot_model(MODEL, show_shapes=True)

X, y, z = load_data(training_images_path, training_masks_path, training_manual_path)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = []

for train_index, test_index in kf.split(X):
    # Separamos X, z (segmentación manual) e y (máscaras visuales FOV)
    X_train, X_test = X[train_index], X[test_index]
    z_train, z_test = z[train_index], z[test_index]
    y_train, y_test = y[train_index], y[test_index]

    X_train, X_test = mask_and_padding(X_train, y_train), mask_and_padding(X_test, y_test)
    z_train, z_test = mask_and_padding(z_train, y_train), mask_and_padding(z_test, y_test)
    
    # Entrenar el modelo (usamos el .fit original de Keras)
    MODEL.fit(X_train, z_train, epochs=10, batch_size=8, verbose=1)
    
    # Predecir sobre X_test
    z_pred = MODEL.predict(X_test)
    z_pred = np.round(z_pred) # Binarizar salidas
    print("Z_PRED_SHAPE:" + str(z_pred.shape))

    # Calcular métrica usando DICE score
    score = dice_score_group(z_train, z_pred, y_train)
    print("DICE Score: " + str(score))
    scores.append(score)

print(f"DICE Scores por iteración: {scores}")
print(f"Media DICE Score: {np.mean(scores)}")