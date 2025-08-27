from src.components.data_downloader import DataDownloader
from src.components.data_ingestion import DataIngestion
from src.components.model_builder import ModelBuilder
from src.components.model_trainer import ModelTrainer
from src.utils.logger import logger
from src.components.evaluate import ModelEvaluator


if __name__ == "__main__":
    try:
        # # #Step 1: Download dataset
        # dataset_url = "https://www.kaggle.com/datasets/faber24/lcc-fasd"
        # downloader = DataDownloader(dataset_url, "data/")
        # logger.info(f"📥 Downloading dataset from {dataset_url} ...")
        # downloader.download()
        # logger.info(f"✅ Dataset downloaded successfully.")


        #Step 2: Dataset base directory
        base_dir = "data/LCC_FASD"
        logger.info(f"Dataset base directory: {base_dir}")

        # Step 3: Data ingestion (train/val/test)
        data_ingestion = DataIngestion(base_dir, img_size=(224, 224), batch_size=32)
        train_gen, val_gen, test_gen, class_weights  = data_ingestion.get_data_generators()
        logger.info("✅ Data generators created successfully.")

        # # Step 4: Build & Train model
        model_builder = ModelBuilder()
        model, base_model = model_builder.build_model(input_shape=(224, 224, 3))
        logger.info("✅ Model built successfully.")

        # # # Step 5: Train model
        save_dir = "models"
        trainer = ModelTrainer(
            model=model,
            base_model=base_model,
            train_gen=train_gen,
            val_gen=val_gen,
            class_weights=class_weights,
            save_dir=save_dir,
            stage1_epochs=5,
            stage2_epochs=30,
        )
        logger.info("🚀 Starting Phase 1 Training...")
        trainer.train_phase1()

        logger.info("🚀 Starting Phase 2 Fine-Tuning...")
        trainer.train_phase2()

        logger.info("✅ Training complete. Best model saved.")

        # Step 6: Evaluate model
        evaluator = ModelEvaluator(model_path=trainer.stage2_path, test_gen=test_gen)
        evaluator.evaluate()
        
    except Exception as e:
        logger.error(f"❌ Training pipeline failed: {e}")
        print(f"[ERROR] Training pipeline failed: {e}")

