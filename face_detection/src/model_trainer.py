import os
from ultralytics import YOLO
from src.logger import logger

class YOLOTrainer:
    def __init__(self, data_yaml, model="yolo11s.pt", epochs=12, imgsz=736, batch=16,
                 optimizer="SGD", lr0=0.001, momentum=0.937, weight_decay=5e-4, patience=15):
        self.data_yaml = data_yaml
        self.model = model
        self.epochs = epochs
        self.imgsz = imgsz
        self.batch = batch
        self.optimizer = optimizer
        self.lr0 = lr0
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.patience = patience


    def train(self):
        """Train YOLO model using Ultralytics."""
        try:
            logger.info("🚀 Starting YOLO training...")
            model = YOLO(self.model)

            results = model.train(
                data=self.data_yaml,
                epochs=self.epochs,
                imgsz=self.imgsz,
                batch=self.batch,
                optimizer=self.optimizer,
                lr0=self.lr0,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
                patience=self.patience,
                device = 'cpu'
                )

            logger.info("✅ Training completed successfully!")
            logger.info(f"📂 Best model saved at: {model.ckpt_path}")
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            raise e

if __name__ == "__main__":
    DATA_YAML_PATH = "data/face-detection-dataset/data.yaml"

    if not os.path.exists(DATA_YAML_PATH):
        logger.error("❌ data.yaml not found. Please run download_data.py first!")
        exit(1)

    trainer = YOLOTrainer(
        data_yaml=DATA_YAML_PATH,
        model="yolo11s.pt",
        epochs=12,
        imgsz=640,
        batch=16,
        optimizer="SGD",
        lr0=0.001,
        momentum=0.937,
        weight_decay=5e-4,
        patience=15
    )
    trainer.train()
