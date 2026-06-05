import os
from keras import mixed_precision
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from UnetModel import build_model
from dotenv import load_dotenv
from sklearn.model_selection import KFold
from dataManagement.DataLoader import load_training_data
from dataManagement.DataGenerator import append_augmented_data
from dataManagement.DataPreprocessing import transform
from Metrics import dice_score_group, dice_score, bce_dice_loss


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

load_dotenv()

DATA_PATH = os.path.abspath("data")
MODELS_PATH = os.path.abspath("models")
TEST_PATH = os.path.join(DATA_PATH, "test")
TRAIN_PATH = os.path.join(DATA_PATH, "training")

HORIZONTAL_UNET_SIZE = int(os.getenv("HORIZONTAL_UNET_SIZE"))
VERTICAL_UNET_SIZE = int(os.getenv("VERTICAL_UNET_SIZE"))

WRITE_MODELS = eval(os.getenv("WRITE_MODELS"))
SHOW_PREDICTIONS_IN_TRAINING = eval(os.getenv("SHOW_PREDICTIONS_IN_TRAINING"))


def train_model():

    x, y, z = load_training_data()
    x, y, z = append_augmented_data(x, y, z)

    kf = KFold(n_splits=5, shuffle=True)
    
    generated_images = []
    scores = []
    fold = 1
    for train_index, test_index in kf.split(x):

        print(f"Training with fold {fold}")

        # Separar x (imágenes de entrada), z (segmentaciones manuales) e y (máscaras)
        x_train, x_test = x[train_index], x[test_index]
        z_train, z_test = z[train_index], z[test_index]
        y_train, y_test = y[train_index], y[test_index]

        x_train, x_test = transform(x_train, masks=y_train), transform(x_test, masks=y_test)
        z_train, z_test = transform(z_train, masks=y_train), transform(z_test, masks=y_test)
        y_test = transform(y_test)
        
        # Definir el modelo
        MODEL = build_model(input_shape=(VERTICAL_UNET_SIZE, HORIZONTAL_UNET_SIZE, 1))
        MODEL.compile(loss=bce_dice_loss, optimizer="Adam", metrics=[dice_score])

        # Entrenar el modelo
        MODEL.fit(x_train, z_train, epochs=100, batch_size=3, verbose=1)
        
        # Predecir sobre x_test
        z_pred = MODEL.predict(x_test)
        z_pred = np.round(z_pred)
        transform_to_img(z_pred, generated_images)

        # Calcular métrica usando DICE score
        scores.append(dice_score_group(z_test, z_pred, y_test))

        MODEL_SAVE_PATH = os.path.join(MODELS_PATH, f"model_{fold}.keras")
        if WRITE_MODELS:
            MODEL.save(MODEL_SAVE_PATH, overwrite=True)
        
        fold += 1

    for i, score in enumerate(scores):
        print(f"Fold {i+1} DICE Score: {score}")
    if SHOW_PREDICTIONS_IN_TRAINING:
        show_generated_images(generated_images)

def transform_to_img(z_pred, generated_images):
    for i in range(z_pred.shape[0]):
        img = z_pred[i, :, :, 0]
        generated_images.append(img)

def show_generated_images(generated_images):
    fig, ax = plt.subplots(5,4, figsize=(10,5))
    for fold in range(5):
        for image in range(4):
            ax[fold,image].imshow(generated_images[image + 4*fold], cmap='gray')
    plt.show()

if __name__ == "__main__":
    train_model()