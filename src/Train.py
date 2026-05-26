from keras.models import Model
import os
import tensorflow as tf
from UnetModel import build_model
from dotenv import load_dotenv
from sklearn.model_selection import cross_validate
from DataGenerator import load_data
from sklearn.metrics import make_scorer
from DiceScore import dice_score
load_dotenv()

ROOT_PATH = os.path.abspath("data")
TEST_PATH = os.path.join(ROOT_PATH, "test")
TRAIN_PATH = os.path.join(ROOT_PATH, "training")

training_images_path = os.path.join(TRAIN_PATH, "images")
training_masks_path = os.path.join(TRAIN_PATH, "mask")
training_manual_path = os.path.join(TRAIN_PATH, "1st_manual")

MODEL = build_model(input_shape=(576, 592, 1))
MODEL.compile(loss="binary_crossentropy", optimizer="Adam", metrics=["accuracy"])

tf.keras.utils.plot_model(MODEL, show_shapes=True)

DICE_SCORE = make_scorer(dice_score)
X, y, z = load_data(training_images_path, training_masks_path, training_manual_path)

resultados_validación_cruzada = cross_validate(MODEL,
                                               X,
                                               z,
                                               scoring=DICE_SCORE,
                                               cv=5)