from src.components.predictor import LivenessPredictor
from src.utils.logger import logger

if __name__ == "__main__":
    try:
        predictor = LivenessPredictor("models/face_liveness_best.h5")
        logger.info("✅ Liveness Predictor initialized successfully.")

        while True:
            print("\nPress 'w' for Webcam, 'p' for Picture, or 's' to Skip...")
            choice = input("Your choice: ").strip().lower()

            if choice == "w":
                while True:
                    cam_input = input(
                        "Select webcam index (0 = built-in, 1 = external, etc.). "
                        "Press Enter to use default [0]: "
                    ).strip()

                    if cam_input == "":
                        cam_index = 0
                        break
                    elif cam_input.isdigit() and int(cam_input) in [0, 1]:
                        cam_index = int(cam_input)
                        break
                    else:
                        print("❌ Invalid input! Enter 0 or 1, or press Enter for default [0].")

                predictor.detect_webcam(cam_index=cam_index)
                break

            elif choice == "p":
                img_path = input("Enter image path: ").strip()
                predictor.detect_faces_and_predict(img_path)
                break

            elif choice == "s":
                print("Exiting program.")
                break

            else:
                print("❌ Invalid choice! Please try again.")
    except Exception as e:
        logger.error(f"❌ Predict pipeline failed: {e}")
        print(f"[ERROR] Something went wrong: {e}")
