import os
import cv2
import numpy as np
from logger import logger
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model

# === CONFIGURATION ===
MODEL_PATH = os.path.join("face_liveness", "models", "face_liveness_best.h5")
CLASS_LABELS = ["Live", "Spoof"]
model = load_model(MODEL_PATH)

def predict_liveness(face,threshold=0.8):
    try:
        # Preprocess image
        img = cv2.resize(face, (224, 224))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Predict using sigmoid output
        prob = model.predict(img_array, verbose=0)[0][0]
        class_idx = 1 if prob >= threshold else 0
        label = CLASS_LABELS[class_idx]

        # If spoof → probability = prob, if live → probability = 1 - prob
        score = prob if class_idx == 1 else 1 - prob
        return label, score
    except Exception as e:
        logger.error(f"❌ Liveness prediction failed: {e}")
        return None, None

if __name__ == "__main__":
    test_img_path = r"user_images\user_subith.jpg"
    test_img = cv2.imread(test_img_path)
    label, conf = predict_liveness(test_img)
    if label is not None:
        print(f"Prediction: {label} ({conf*100:.2f}%)")
    else:
        print("Prediction failed!")
