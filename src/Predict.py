import pickle
from keras.saving import load_model
import os
import cv2
from DataGenerator import load_data_test
from DiceScore import dice_score_group
from Train import mask_and_padding

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

MODELS_PATH = os.path.abspath("models")
MODEL_FILE = "model_5.keras"
MODEL_PATH = os.path.join(MODELS_PATH, MODEL_FILE)

DATA_PATH = os.path.abspath("data")
TEST_PATH = os.path.join(DATA_PATH, "test")
GENERATED_PATH = os.path.join(DATA_PATH, "generated")

def load_data_test_paths():
    test_images_path = os.path.join(TEST_PATH, "images")
    test_masks_path = os.path.join(TEST_PATH, "mask")
    test_1st_manual_path = os.path.join(TEST_PATH, "1st_manual")
    test_2nd_manual_path = os.path.join(TEST_PATH, "2nd_manual")
    return test_images_path, test_masks_path, test_1st_manual_path, test_2nd_manual_path

def load_data():
    test_images_path, test_masks_path, test_1st_manual_path, test_2nd_manual_path = load_data_test_paths()
    X_test, y_test, z_test_1, z_test_2 = load_data_test(test_images_path, test_masks_path, test_1st_manual_path, test_2nd_manual_path)
    return X_test, y_test, z_test_1, z_test_2

def prediction_scores(z_true_1, z_true_2, z_pred, mask):
    score1 = dice_score_group(z_true_1, z_pred, mask)
    score2 = dice_score_group(z_true_2, z_pred, mask)
    return score1, score2, (score1 + score2) / 2

def save_images(z_pred):
    for i in range(z_pred.shape[0]):
        img = z_pred[i, :, :, 0] * 255
        path = os.path.join(GENERATED_PATH, f"generated_image_{i}.png")
        cv2.imwrite(path, img)

def predict():
    MODEL = load_model(MODEL_PATH)
    X_test, y_test, z_test_1, z_test_2 = load_data()
    
    X_test = mask_and_padding(X_test, y_test)
    z_test_1 = mask_and_padding(z_test_1, y_test)
    z_test_2 = mask_and_padding(z_test_2, y_test)
    y_test = mask_and_padding(y_test, y_test)
    
    z_pred = MODEL.predict(X_test)

    score1, score2, total_score = prediction_scores(z_test_1, z_test_2, z_pred, y_test)
    print("DICE Score 1st manual: " + str(score1))
    print("DICE Score 2nd manual: " + str(score2))
    print("DICE Score average: " + str(total_score))
    save_images(z_pred)

if __name__ == "__main__":
    predict()