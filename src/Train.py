import os
from keras.models import Model
from keras import mixed_precision
import tensorflow as tf
import matplotlib.pyplot as plt

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Ignorar logs informativos de CUDA y TF, mostrar solo errores fatales

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
from DiceScore import dice_score_group, dice_score, dice_score_loss, bce_dice_loss

load_dotenv()

DATA_PATH = os.path.abspath("data")
MODELS_PATH = os.path.abspath("models")
TEST_PATH = os.path.join(DATA_PATH, "test")
TRAIN_PATH = os.path.join(DATA_PATH, "training")
HORIZONTAL_PADDING_SIZE = int(os.getenv("HORIZONTAL_PADDING_SIZE", 576))
VERTICAL_PADDING_SIZE = int(os.getenv("VERTICAL_PADDING_SIZE", 592))
OVERWRITE_MODELS = True

def load_data_training_paths():
     training_images_path = os.path.join(TRAIN_PATH, "images")
     training_masks_path = os.path.join(TRAIN_PATH, "mask")
     training_manual_path = os.path.join(TRAIN_PATH, "1st_manual")
     return training_images_path, training_masks_path, training_manual_path

generated_images = []

def train_model():

    MODEL = build_model(input_shape=(VERTICAL_PADDING_SIZE, HORIZONTAL_PADDING_SIZE, 1))
    MODEL.compile(loss=bce_dice_loss, optimizer="Adam", metrics=[dice_score])
    
    training_images_path, training_masks_path, training_manual_path = load_data_training_paths()

    X, y, z = load_data(training_images_path, training_masks_path, training_manual_path)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    scores = []

    fold = 1
    for train_index, test_index in kf.split(X):
        # Separamos X, z (segmentación manual) e y (máscaras visuales FOV)
        X_train, X_test = X[train_index], X[test_index]
        z_train, z_test = z[train_index], z[test_index]
        y_train, y_test = y[train_index], y[test_index]

        X_train, X_test = mask_and_padding(X_train, y_train), mask_and_padding(X_test, y_test)
        z_train, z_test = mask_and_padding(z_train, y_train), mask_and_padding(z_test, y_test)
        y_train, y_test = mask_and_padding(y_train, y_train), mask_and_padding(y_test, y_test)
        
        # Entrenar el modelo
        MODEL.fit(X_train, z_train, epochs=50, batch_size=6, verbose=1)
        
        # Predecir sobre X_test
        z_pred = MODEL.predict(X_test)
        transform_to_img(z_pred)

        # Calcular métrica usando DICE score
        score = dice_score_group(z_test, z_pred, y_test)
        print("DICE Score: " + str(score))
        scores.append(score)

        MODEL_SAVE_PATH = os.path.join(MODELS_PATH, f"model_{fold}.keras")
        MODEL.save(MODEL_SAVE_PATH, overwrite=OVERWRITE_MODELS)
        
        fold += 1

    print(f"Media DICE Score: {np.mean(scores)}")
    show_generated_images()

def transform_to_img(z_pred):
    for i in range(z_pred.shape[0]):
        img = z_pred[i, :, :, 0]
        print(img.max())
        generated_images.append(img)
    print("Generated images size:", len(generated_images))

def show_generated_images():
    fig, ax = plt.subplots(5,4, figsize=(10,5))
    for fold in range(5):
        for image in range(4):
            ax[fold,image].imshow(generated_images[image + 4*fold], cmap='gray')
    plt.show()

if __name__ == "__main__":
    train_model()