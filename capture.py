import cv2
from PIL import Image

def capture_face():
    cap = cv2.VideoCapture(0)
    print("Press SPACE to capture image")

    while True:
        ret, frame = cap.read()
        cv2.imshow('Capture Face', frame)
        if cv2.waitKey(1) & 0xFF == ord(' '):
            image = frame.copy()
            break

    cap.release()
    cv2.destroyAllWindows()
    return Image

