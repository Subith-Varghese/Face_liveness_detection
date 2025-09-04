# 🧑‍💻 Face Recognition with Liveness Detection & Attendance System

This project integrates:
- **YOLOv11** → Face detection  
- **MobileNetV2 (CNN)** → Face liveness detection (anti-spoofing)  
- **FaceNet (InceptionResnetV1)** → Face recognition    

It ensures that only **live human faces** are recognized and prevents spoofing attempts (photos, videos, printouts). Attendance is automatically logged once a registered user is recognized.

---
```
## 📂 Project Structure

project/
│
├── face_detection/ # YOLOv11 face detection
│ └── runs/detect/train/weights/best.pt
│
├── face_liveness/ # CNN liveness model
│ └── models/face_liveness_best.h5
│
├── user_images/ # Saved face images during registration
│
├── logs/ # Log files
│
├── attendance.csv # Attendance log (auto-created)
├── face_db.pkl # Face embeddings database (auto-created)
│
├── face_detect.py # Face detection with YOLOv11
├── face_liveness.py # Liveness detection (CNN)
├── face_recognize.py # Face recognition (FaceNet)
├── register.py # Register new users
├── main.py # Main pipeline (detection + liveness + recognition + logging)
├── login_attendance.py # Attendance logging
├── logger.py # Logging setup
└── README.md # Documentation
```

---

## Workflow

### 1. Face Detection (YOLOv11)
- Detects faces in real-time using the trained YOLOv11 model (`best.pt`).
- Crops the detected faces for further processing.

### 2. Liveness Detection (MobileNetV2)
- Two-stage training:
  - Stage 1 → Train top layers (frozen base).
  - Stage 2 → Fine-tune last layers.
- It produces face_liveness_best.h5.
- `face_liveness_best.h5` determines if the detected face is:
  - ✅ **Live** → Proceed to recognition  
  - ❌ **Spoof (photo/video/print)** → Rejected immediately  

### 3. **Face Recognition (FaceNet)**
- Extracts embeddings (512-d vectors) using **InceptionResnetV1 (FaceNet)**.
- Compares embeddings against the **face database (`face_db.pkl`)** using cosine similarity.
- If similarity > threshold (default 0.6), the user is recognized.

### 4. **Attendance Logging**
- Once a **live & recognized** user is found:
  - Name and timestamp are written to `attendance.csv`.
  - Prevents duplicate logging in the same session.

---

```
           ┌─────────────┐
           │   YOLOv11   │
           │ Face Detect │
           └──────┬──────┘
                  │ Cropped Face
                  ▼
           ┌─────────────┐
           │  CNN Model  │
           │ LivenessChk │
           └──────┬──────┘
         Live     │     Spoof
          │       │
          ▼       ▼
   ┌───────────┐  ✖ SPOOF REJECTED
   │  FaceNet  │
   │ Embedding │
   └─────┬─────┘
         │ Compare (Cosine Similarity)
         ▼
   ┌─────────────┐
   │ Recognition │───► Attendance Logged (CSV)
   └─────────────┘


```

## 🛠️ Installation

```bash
# Clone repository
git clone https://github.com/Subith-Varghese/Face_liveness_detection.git
cd face-recognition-attendance

# Create virtual environment
python -m venv venv
venv\Scripts\activate     

# Install dependencies
pip install -r requirements.txt
```

### Dependencies include:

- ultralytics (YOLOv8/YOLOv11 support)
- torch, torchvision, facenet-pytorch
- tensorflow / keras
- opencv-python
- numpy, scikit-learn

--- 
## 🚀 Usage
### 1. Register a New User

```python register.py```

- Choose to capture face from camera or provide an image.
- Saves cropped face in user_images/.
- Stores face embedding in face_db.pkl.

### 2. Run Face Recognition with Liveness Detection
``` python main.py```

- Opens webcam feed.
- Detects multiple faces.
- Runs liveness detection → recognition → attendance logging.
- Press q to exit.

### 3. Check Attendance Logs

attendance.csv

## Example Workflow

1. Register user Alice via webcam.
2. Run main.py → Alice’s face appears → CNN validates liveness → FaceNet matches → Attendance logged.
3. If a spoof attempt (photo/video) is shown → CNN rejects with "SPOOF DETECTED".
4. Logs are stored in logs/ and attendance.csv.

--- 
## Key Features

✅ Multi-Face Detection (handles multiple people simultaneously).
✅ Anti-Spoofing (CNN liveness detection).
✅ Real-Time Recognition (using YOLOv11 + FaceNet).
✅ Automatic Attendance Logging.
✅ Scalable User Registration (database grows dynamically).


