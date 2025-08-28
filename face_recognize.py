from torchvision import transforms   
import cv2
import torch
from facenet_pytorch import InceptionResnetV1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)


# Image Preprocessing
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])     

def face_recognize_model(image):
    # Step 4: Convert to RGB and preprocess
    print("[STEP 3] Preprocessing face for recognition...")
    face_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_tensor = transform(face_rgb).unsqueeze(0).to(device)
    print(f"✅ Face tensor shape: {face_tensor.shape}")
        
    # Step 5: Create face embedding
    print("Creating face signature...")
    embedding = model(face_tensor).detach().cpu()
    print(f"✅ Embedding created. Shape: {embedding.shape}")
    return embedding