from ultralytics import YOLO
import os

# Since you're inside face_detection, set paths relative to this folder
model_path = os.path.join("runs", "detect", "train", "weights", "best.pt")
data_yaml = os.path.join("data", "face-detection-dataset", "data.yaml")

model =  YOLO(model_path)

def evaluate_model():
    metrics = model.val(
    data=data_yaml,  
    imgsz=640,                                        
    batch=16)

    # Print evaluation metrics
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"mAP50: {metrics.box.map50:.4f}")

if __name__ == '__main__':
    evaluate_model()


