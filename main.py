import os
import torch
import pickle
import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from login_attendance import log_attendance
from face_detect import face_detect_yolo
from face_liveness import predict_liveness
from face_recognize import face_recognize_model


def recognize_faces():
    
    # Load known faces
    known_faces = {}
    if os.path.exists('face_db.pkl'):
        with open('face_db.pkl', 'rb') as f:
            known_faces = pickle.load(f)
            known_names = list(known_faces.keys())
            known_embeddings = torch.cat([emb for emb in known_faces.values()])
    else:
        known_names = []
        known_embeddings = torch.empty((0, 512))  # Empty tensor
        print("No known face database found. Starting with empty database.")

    already_logged = set()

    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("🔍 Starting Multi-Face Recognition with Liveness Detection... Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        final_boxes = face_detect_yolo(frame)

        if final_boxes is not None:
            embeddings = []
            for x1, y1, x2, y2, conf, cls in final_boxes:
                x1, y1, x2, y2 = map(int,[x1, y1, x2, y2])
                cropped_face = frame[y1:y2, x1:x2]

                # Step 1: Liveness Check
                label,scorelive = predict_liveness(cropped_face)  # Must return True or False

                if label == 'Live':
                    # Step 2: Face embedding
                    embedding = face_recognize_model(cropped_face)
                    embeddings.append((embedding, (x1, y1, x2, y2)))
                else:
                    # Draw red box for spoof
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, f"SPOOF DETECTED score : {scorelive}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # Step 3: Recognition for live faces only
            for emb, (x1, y1, x2, y2) in embeddings:
                if known_embeddings.shape[0] > 0:
                    sims = cosine_similarity(emb.numpy(), known_embeddings.numpy())[0]
                    best_match = np.argmax(sims)
                    similarity = sims[best_match]
                    name = known_names[best_match] if similarity > 0.6 else "Unknown"
                else:
                    name = "No reference face"
                    similarity = 0.0

                color = (0, 255, 0) if name != "Unknown" and name != "No reference face" else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{name} ({similarity:.2f})", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                if name != "Unknown" and name not in already_logged:
                    log_attendance(name)
                    already_logged.add(name)

        cv2.imshow("Multi-Face Recognition with Liveness Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    recognize_faces()
