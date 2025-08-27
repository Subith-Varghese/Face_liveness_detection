from tensorflow.keras import Sequential, layers
from tensorflow.keras.applications import MobileNetV2
from src.utils.logger import logger


class ModelBuilder:
    @staticmethod
    def build_model(input_shape=(224, 224, 3)):
        try:
            logger.info("🔧 Building the MobileNetV2 model...")
            
            # Load pre-trained MobileNetV2 as base model
            base_model = MobileNetV2(
                weights="imagenet",
                include_top=False,
                input_shape=input_shape
            )
            base_model.trainable = False  # Freeze base layers initially

            # Build full model
            model = Sequential([
                base_model,
                layers.Conv2D(32, (3, 3), activation="relu"),
                layers.Dropout(0.3),
                layers.GlobalAveragePooling2D(),
                layers.Dense(256, activation="relu"),
                layers.BatchNormalization(),
                layers.Dropout(0.3),
                layers.Dense(1, activation="sigmoid")
            ])
            logger.info("✅ Model built successfully.")
            return model,base_model

        except Exception as e:
            logger.error(f"❌ Failed to build model: {e}")
            raise e
