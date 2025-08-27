import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from src.utils.logger import logger
from sklearn.utils.class_weight import compute_class_weight
import numpy as np


class DataIngestion:
    def __init__(self, base_dir="data/LCC_FASD", img_size=(224, 224), batch_size=32):
        self.base_dir = base_dir
        self.train_dir = os.path.join(base_dir, "LCC_FASD_training")
        self.val_dir = os.path.join(base_dir, "LCC_FASD_development")
        self.test_dir = os.path.join(base_dir, "LCC_FASD_evaluation")
        self.img_size = img_size
        self.batch_size = batch_size

    def get_data_generators(self):
        try:
            # ✅ Data augmentation for training
            train_aug = ImageDataGenerator(
                preprocessing_function=preprocess_input,
                rotation_range=20,
                zoom_range=0.2,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.15,
                horizontal_flip=True,
                fill_mode="nearest"
            )

            # ✅ Validation & Test → No augmentation
            val_aug = ImageDataGenerator(preprocessing_function=preprocess_input)

            # ✅ Training data generator
            train_gen = train_aug.flow_from_directory(
                self.train_dir,
                target_size=self.img_size,
                batch_size=self.batch_size,
                class_mode="binary"  
            )
            logger.info("✅ Training data generator created.")

            # ✅ Validation data generator
            val_gen = val_aug.flow_from_directory(
                self.val_dir,
                target_size=self.img_size,
                batch_size=self.batch_size,
                class_mode="binary"
            )
            logger.info("✅ Validation data generator created.")

            # ✅ Test data generator
            test_gen = val_aug.flow_from_directory(
                self.test_dir,
                target_size=self.img_size,
                batch_size=self.batch_size,
                class_mode="binary",
                shuffle=False
            )
            logger.info("✅ Test data generator created.")

            print("train_gen.class_indices",train_gen.class_indices)
            classes = np.unique(train_gen.classes)

            class_weights = compute_class_weight(
                class_weight="balanced",
                classes=classes,
                y=train_gen.classes
            )

            class_weights = dict(zip(classes, class_weights))
            print("Class Weights:", class_weights)

            return train_gen, val_gen, test_gen, class_weights

        except Exception as e:
            logger.error(f"❌ Failed to create data generators: {e}")
            raise e
