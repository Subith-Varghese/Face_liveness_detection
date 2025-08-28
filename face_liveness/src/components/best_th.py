import numpy as np
from sklearn.metrics import roc_curve
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from src.components.data_ingestion import DataIngestion
import matplotlib.pyplot as plt

base_dir = "data/LCC_FASD"
MODEL_PATH = "models/face_liveness_best.h5"
model = load_model(MODEL_PATH)

if __name__ == "__main__":

    data_ingestion = DataIngestion(base_dir, img_size=(224, 224), batch_size=32)
    train_gen, val_gen, test_gen, class_weights  = data_ingestion.get_data_generators()
    print("✅ Data generators created successfully.")

    # Combine all generators
    all_gens = [train_gen, val_gen, test_gen]

    probs = []
    y_true = []

    for gen in all_gens:
        gen.reset()
        for i in range(len(gen)):
            X_batch, y_batch = gen[i]
            batch_probs = model.predict(X_batch, verbose=0)[:, 0]  # sigmoid output
            probs.extend(batch_probs)
            y_true.extend(y_batch)

    probs = np.array(probs)
    y_true = np.array(y_true)

    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, probs)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_threshold = thresholds[best_idx]

    print(f"✅ Optimal threshold determined: {best_threshold:.3f}")

    # Optional: ROC curve plot
    plt.plot(fpr, tpr, label="ROC curve")
    plt.plot([0,1], [0,1], linestyle="--", color="gray")
    plt.scatter(fpr[best_idx], tpr[best_idx], color="red", label=f"Best Threshold: {best_threshold:.3f}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Face Liveness Detection")
    plt.legend()
    plt.show()
