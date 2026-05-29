from keras.saving import load_model
import os
import cv2
from DataGenerator import load_data_test
from DiceScore import dice_score_group
from Train import transform
from DataPostprocessing import detransform


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

MODELS_PATH = os.path.abspath("models")
MODEL_FILE = "model_5.keras"
MODEL_PATH = os.path.join(MODELS_PATH, MODEL_FILE)

DATA_PATH = os.path.abspath("data")
TEST_PATH = os.path.join(DATA_PATH, "test")
GENERATED_PATH = os.path.join(DATA_PATH, "generated")
WRITE_IMAGES = True

def get_images_sizes(images):
    images_sizes = []
    for i in range(images.shape[0]):
        image = images[i]
        images_sizes.append((image.shape[0], image.shape[1]))
    return images_sizes

def get_prediction_scores(z_true_1, z_true_2, z_pred, mask):
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

    X_test, y_test, z_test_1, z_test_2 = load_data_test()

    images_sizes = get_images_sizes(X_test)
    
    X_test = transform(X_test, masks=y_test)
    z_test_1 = transform(z_test_1, masks=y_test)
    z_test_2 = transform(z_test_2, masks=y_test)
    y_test = transform(y_test)
    
    z_pred = MODEL.predict(X_test)

    score1, score2, total_score = get_prediction_scores(z_test_1, z_test_2, z_pred, y_test)
    print("DICE Score 1st manual: " + str(score1))
    print("DICE Score 2nd manual: " + str(score2))
    print("DICE Score average: " + str(total_score))

    z_pred_detransformed = detransform(z_pred, original_image_sizes=images_sizes)

    if WRITE_IMAGES:
        save_images(z_pred_detransformed)

if __name__ == "__main__":
    predict()