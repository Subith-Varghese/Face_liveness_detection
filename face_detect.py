import os
from ultralytics import YOLO
import cv2
import numpy as np

yolo_model_path = os.path.join("face_detection","runs", "detect", "train", "weights", "best.pt")
model = YOLO(yolo_model_path)


def face_detect_yolo(image,register=False):
    
    # Determine if 'image' is a path or already a loaded image
    if isinstance(image, str):
        img = cv2.imread(image)
        if img is None:
            print(f"❌ Error: Could not load image from {image}")
            return None
    elif isinstance(image, np.ndarray):
        img = image.copy()
    else:
        print("❌ Error: 'image' must be a file path or np.ndarray")
        return None

    # Run YOLO prediction
    results = model.predict(img)
    result = results[0]
    print("\n=== YOLOv8 PREDICTION RESULTS ===")

    boxes = result.boxes
    names = result.names

    cropped_face_list = []  # Store all cropped faces
    # Convert results to a list for manual filtering
    filtered_boxes = []

    for box in boxes:
        conf = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cls = int(box.cls[0])
        filtered_boxes.append((x1, y1, x2, y2, conf, cls))

    # Sort detections by confidence (highest first)
    filtered_boxes.sort(key=lambda x: x[4], reverse=True)

    # Apply simple Non-Maximum Suppression (NMS)
    final_boxes = []
    iou_threshold = 0.4

    def iou(box1, box2):
        # Calculate IoU for two boxes
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - inter_area
        return inter_area / union_area if union_area > 0 else 0

    for box in filtered_boxes:
        if all(iou(box, kept) < iou_threshold for kept in final_boxes):
            final_boxes.append(box)
            
    # If no faces found
    if len(final_boxes) == 0:
        print("❌ No face detected!")
        return None  # No face detected

    
    if register:
        # Return the **first detected cropped face**
        x1, y1, x2, y2, conf, cls = final_boxes[0]
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cropped_face = img[y1:y2, x1:x2]
        return cropped_face
    
    # Otherwise, return only the final face bounding boxes
    return final_boxes





if __name__ == "__main__": 
    face  = face_detect_yolo("IMG_20230803_162015 (1)(1).jpg",
                     True)
    if face is not None:
        print("🎯 Face detection successful!")
    else:
        print("⚠️ No face detected!")