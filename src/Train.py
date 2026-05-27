import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Ignorar logs informativos de CUDA y TF, mostrar solo errores fatales

from keras.models import Model
from keras import mixed_precision
import tensorflow as tf

# Habilitar precisión mixta FP16 para reducir el consumo a la mitad
mixed_precision.set_global_policy('mixed_float16')

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

from UnetModel import build_model
from dotenv import load_dotenv
from sklearn.model_selection import KFold, cross_validate
from DataGenerator import load_data
import numpy as np
from DataPreprocesing import mask_and_padding
from DiceScore import dice_score_group, dice_score

load_dotenv()

ROOT_PATH = os.path.abspath("data")
TEST_PATH = os.path.join(ROOT_PATH, "test")
TRAIN_PATH = os.path.join(ROOT_PATH, "training")
HORIZONTAL_PADDING_SIZE = int(os.getenv("HORIZONTAL_PADDING_SIZE", 576))
VERTICAL_PADDING_SIZE = int(os.getenv("VERTICAL_PADDING_SIZE", 592))

def load_paths():
     training_images_path = os.path.join(TRAIN_PATH, "images")
     training_masks_path = os.path.join(TRAIN_PATH, "mask")
     training_manual_path = os.path.join(TRAIN_PATH, "1st_manual")
     return training_images_path, training_masks_path, training_manual_path

def train_model():
    training_images_path, training_masks_path, training_manual_path = load_paths()
    MODEL = build_model(input_shape=(VERTICAL_PADDING_SIZE, HORIZONTAL_PADDING_SIZE, 1))
    MODEL.compile(loss="binary_crossentropy", optimizer="Adam", metrics=[dice_score])

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
        y_train, y_test = mask_and_padding(y_train, y_train), mask_and_padding(y_test, y_test)
        
        # Entrenar el modelo (usamos el .fit original de Keras)
        MODEL.fit(X_train, z_train, epochs=5, batch_size=6, verbose=1)
        
        # Predecir sobre X_test
        z_pred = MODEL.predict(X_test)
        z_pred = np.round(z_pred) # Binarizar salidas
        print("Z_PRED_SHAPE:" + str(z_pred.shape))

        # Calcular métrica usando DICE score
        score = dice_score_group(z_test, z_pred, y_test)
        print("DICE Score: " + str(score))
        scores.append(score)

    print(f"Media DICE Score: {np.mean(scores)}")

if __name__ == "__main__":
    train_model()