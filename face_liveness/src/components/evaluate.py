import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model
from src.utils.logger import logger


class ModelEvaluator:
    def __init__(self, model_path, test_gen):
        """
        Initialize evaluator.
        :param model_path: Path to the trained model (.h5 file)
        :param test_gen: Test data generator
        """
        self.model_path = model_path
        self.test_gen = test_gen

        # Check if model exists
        if not os.path.exists(self.model_path):
            logger.error(f"❌ Model file not found at: {self.model_path}")
            raise FileNotFoundError(f"Model file not found at {self.model_path}")
        else:
            logger.info(f"✅ Model found at: {self.model_path}")

    def evaluate(self):
        """
        Evaluate the model on the test dataset.
        """
        try:
            # Load the trained model
            logger.info(f"📥 Loading model from: {self.model_path}")
            model = load_model(self.model_path)

            # Evaluate model performance
            loss, accuracy, precision, recall = model.evaluate(self.test_gen, verbose=1)
            logger.info("📊 Model Evaluation Results:")
            print("\n=== Test Set Metrics ===")
            print(f"Loss      : {loss:.4f}")
            print(f"Accuracy  : {accuracy:.4f}")
            print(f"Precision : {precision:.4f}")
            print(f"Recall    : {recall:.4f}")

            # Predict on test data
            logger.info("🔍 Generating predictions...")
            y_pred_prob = model.predict(self.test_gen, verbose=1)
            y_pred = (y_pred_prob > 0.5).astype(int).ravel()
            y_true = self.test_gen.classes

            # Classification report
            logger.info("📝 Generating classification report...")
            target_names = list(self.test_gen.class_indices.keys())
            print("\n=== Classification Report ===")
            print(classification_report(y_true, y_pred, target_names=target_names, digits=4))

            # Confusion Matrix
            logger.info("📌 Plotting confusion matrix...")
            cm = confusion_matrix(y_true, y_pred)
            plt.figure(figsize=(6, 5))
            sns.heatmap(
                cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=target_names, yticklabels=target_names
            )
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.title("Confusion Matrix")
            plt.show()

            logger.info("✅ Model evaluation completed successfully.")

        except Exception as e:
            logger.error(f"❌ Evaluation failed: {e}")
            raise e
