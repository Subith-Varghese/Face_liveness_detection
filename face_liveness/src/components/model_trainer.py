from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.models import load_model
from src.utils.logger import logger
import os

class ModelTrainer:
    def __init__(self, model, base_model, train_gen, val_gen, class_weights,
                 save_dir="models", stage1_epochs=5, stage2_epochs=30):
        self.model = model
        self.base_model = base_model
        self.train_gen = train_gen
        self.val_gen = val_gen
        self.class_weights = class_weights
        self.stage1_epochs = stage1_epochs
        self.stage2_epochs = stage2_epochs

        os.makedirs(save_dir, exist_ok=True)
        self.stage1_path = os.path.join(save_dir, "face_liveness_stage1.h5")
        self.stage2_path = os.path.join(save_dir, "face_liveness_best.h5")

    def compile_model(self, learning_rate):
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss="binary_crossentropy",
            metrics=["accuracy", Precision(name="precision"), Recall(name="recall")]
        )

    def train_phase1(self):
        logger.info("🚀 Starting Phase 1: Training top layers (frozen backbone)...")
        self.compile_model(learning_rate=1e-4)

        checkpoint = ModelCheckpoint(self.stage1_path, monitor="val_loss", save_best_only=True, verbose=1)
        early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1)
        reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2, min_lr=5e-7, verbose=1)

        history = self.model.fit(
            self.train_gen,
            validation_data=self.val_gen,
            epochs=self.stage1_epochs,
            class_weight=self.class_weights,
            callbacks=[checkpoint, early_stop, reduce_lr]
        )
        logger.info("✅ Phase 1 complete.")
        return history

    def train_phase2(self):
        logger.info("🚀 Starting Phase 2: Fine-tuning last layers...")
        if os.path.exists(self.stage1_path):
            self.model = load_model(self.stage1_path)
        else:
            logger.warning("⚠️ Best Phase 1 model not found! Using current in-memory model.")


        # Unfreeze layers from a specific block
        freeze_before = "block_16_expand"
        set_trainable = False
        for layer in self.base_model.layers:
            if layer.name == freeze_before:
                set_trainable = True
            layer.trainable = set_trainable

        # Compile with lower learning rate
        self.compile_model(learning_rate=5e-5)

        checkpoint = ModelCheckpoint(self.stage2_path, monitor="val_loss", save_best_only=True, verbose=1)
        early_stop = EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True, verbose=1)
        reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=2, min_lr=5e-7, verbose=1)

        history = self.model.fit(
            self.train_gen,
            validation_data=self.val_gen,
            epochs=self.stage2_epochs,
            class_weight=self.class_weights,
            callbacks=[checkpoint, early_stop, reduce_lr]
        )
        logger.info("✅ Phase 2 complete. Model saved successfully.")
        return history
