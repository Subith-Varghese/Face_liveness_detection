import os
import pickle
import cv2
from facenet_pytorch import InceptionResnetV1
import torch
from logger import logger
from face_detect import face_detect_yolo
from torchvision import transforms
from face_recognize import face_recognize_model

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# face_recognise_model = InceptionResnetV1(pretrained='vggface2').eval().to(device)


# Image Preprocessing
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def capture_face():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    print("Press SPACE to capture image")

    while True:
        ret, frame = cap.read()
        cv2.imshow('Capture Face', frame)
        if cv2.waitKey(1) & 0xFF == ord(' '):
            image = frame.copy()
            break

    cap.release()
    cv2.destroyAllWindows()
    return image


def register_user(name, image_path='cam', db_path='face_db.pkl'):
    """
    Register a new user's face either from camera or image file
    Args:
        name (str): Name of the person to register
        image_path (str): 'cam' to use camera, or path to an image file
        db_path (str): Path to save face database
    """
    print(f"\nStarting registration for: {name}")
    
    try:
        # Step 1: Get the image
        print("[STEP 1] Capturing or loading image...")
        if image_path == 'cam':
            print("Please look at the camera and press SPACE when ready...")
            img = capture_face()  # Uses webcam
        else:
            print(f"Loading image from: {image_path}")
            img = cv2.imread(image_path)
        # Step 2: Detect face
        print("Detecting face...")
        face = face_detect_yolo(img,register=True)
        if face is None:
            print("Error: No face detected! Please try again.")
            return False
        
        # Step 3: save the user face image
        save_img_dir = "user_images"
        os.makedirs(save_img_dir, exist_ok=True)
        save_path = os.path.join(save_img_dir, f"user_{name}.jpg")
        cv2.imwrite(save_path, face)
        print(f"saved image to {save_path}")

        embedding = face_recognize_model(face)

        # # Step 4: Convert to RGB and preprocess
        # print("[STEP 3] Preprocessing face for recognition...")
        # face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        # face_tensor = transform(face_rgb).unsqueeze(0).to(device)
        # print(f"✅ Face tensor shape: {face_tensor.shape}")
            
        # # Step 5: Create face embedding
        # print("Creating face signature...")
        # embedding = face_recognise_model(face_tensor).detach().cpu()
        # print(f"✅ Embedding created. Shape: {embedding.shape}")
        
        # Step 6: Load or create database
        if os.path.exists(db_path):
            with open(db_path, 'rb') as f:
                db = pickle.load(f)
            print(f"✅ Loaded existing database with {len(db)} users")
        else:
            print(" No database found. Creating new one...")
            db = {}
        
        # step 7 : Save embedding
        print(f"[STEP 6] Saving embedding for user: {name}")
        db[name] = embedding
        with open(db_path, 'wb') as f:
            pickle.dump(db, f)
        
        print(f"SUCCESS: {name} has been registered!")
        return True
        
    except Exception as e:
        logger.error(f"ERROR: Registration failed - {str(e)}")
        print(f"❌ Exception: {str(e)}")
        return False

if __name__ == "__main__":
    print("\n=== Face Registration System ===")
    
    # Get user input
    name = input("Enter person's name: ").strip()
    while True:
        choice = input("Register from (1) Camera or (2) Image file? [1/2]: ").strip()
        
        if choice == '1':
            success = register_user(name)
            break
        elif choice == '2':
            image_path = input("Enter image file path: ").strip()
            if os.path.exists(image_path):
                success = register_user(name, image_path)
                break
            else:
                print("File not found! Please try again.")
        else:
            print("Please enter 1 or 2")
    
    if not success:
        print("Registration unsuccessful. Please try again with better lighting/photo.")

