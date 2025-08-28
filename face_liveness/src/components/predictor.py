import os
import cv2
import numpy as np
import torch
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from src.utils.logger import logger
from facenet_pytorch import MTCNN
from tensorflow.keras.models import load_model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"[INFO] MTCNN running on device: {device}")

# Initialize MTCNN once with GPU if available
mtcnn_detector = MTCNN(keep_all=True, device=device)


class LivenessPredictor:
    def __init__(self, model_path="models/face_liveness_best.h5"):
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model not found: {model_path}")

            # Load trained model
            self.model = load_model(model_path)
            self.mtcnn = mtcnn_detector
            self.class_labels = ["Live", "Spoof"]  # Binary labels only
            logger.info(f"✅ Liveness Predictor initialized with model: {model_path}")

        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise e

    def predict_liveness(self, face,threshold=0.746):
        try:
            # Preprocess image
            img = cv2.resize(face, (224, 224))
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)

            # Predict using sigmoid output
            prob = self.model.predict(img_array, verbose=0)[0][0]
            class_idx = 1 if prob >= threshold else 0
            label = self.class_labels[class_idx]

            # If spoof → probability = prob, if live → probability = 1 - prob
            score = prob if class_idx == 1 else 1 - prob
            return label, score
        except Exception as e:
            logger.error(f"❌ Liveness prediction failed: {e}")
            return None, None

    def detect_faces_and_predict(self, img_path, path=True, show=True, resize_factor=None):
        try:
            # Load image
            if path:
                img_bgr = cv2.imread(img_path)
                if img_bgr is None:
                    logger.error(f"[ERROR] Could not read image from {img_path}")
                    return
            else:
                img_bgr = img_path

            # Resize image if required
            if resize_factor is not None:
                small_frame = cv2.resize(img_bgr, (0, 0), fx=resize_factor, fy=resize_factor)
                img_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            else:
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            # Detect faces using MTCNN
            boxes, _ = self.mtcnn.detect(img_rgb)
            h, w, _ = img_bgr.shape
            margin_ratio = 0.25

            if boxes is not None:
                if resize_factor is not None:
                    boxes = boxes / resize_factor

                for box in boxes:
                    x1, y1, x2, y2 = box.astype(int)
                    face_w = x2 - x1
                    face_h = y2 - y1

                    # Add margins for better cropping
                    margin_x = int(face_w * margin_ratio)
                    margin_y = int(face_h * margin_ratio)
                    x1_pad = max(0, x1 - margin_x)
                    y1_pad = max(0, y1 - margin_y)
                    x2_pad = min(w, x2 + margin_x)
                    y2_pad = min(h, y2 + margin_y)

                    # Crop face
                    face = img_bgr[y1_pad:y2_pad, x1_pad:x2_pad]

                    label, prob = self.predict_liveness(face)
                    if label is None:
                        logger.warning(f"⚠ Skipping face at {x1, y1, x2, y2} due to prediction error.")
                        continue

                    logger.info(f"Predicted {label} with probability {prob:.2f} at {x1, y1, x2, y2}")

                    # Draw bounding box and label
                    color = (0, 255, 0) if label == "Live" else (0, 0, 255)
                    cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        img_bgr,
                        f"{label}: {prob:.2f}",
                        (x1, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2,
                    )

            # Show image if requested
            if show:
                cv2.imshow("Face Liveness Detection", img_bgr)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                return img_bgr

        except Exception as e:
            logger.error(f"❌ Face detection failed: {e}")
            return None

    def detect_webcam(self, cam_index=0):
        try:
            cap = cv2.VideoCapture(cam_index)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            logger.info("[INFO] Webcam started. Press 'q' to quit.")

            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Failed to read frame from webcam")
                    break

                annotated_frame = self.detect_faces_and_predict(frame, path=False, show=False, resize_factor=0.5)
                if annotated_frame is not None:
                    cv2.imshow("Face Liveness Detection", annotated_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()
            logger.info("[INFO] Webcam closed successfully.")

        except Exception as e:
            logger.error(f"❌ Webcam detection failed: {e}")

