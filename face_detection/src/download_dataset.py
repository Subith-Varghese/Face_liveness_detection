import os
import opendatasets as od
from src.logger import logger

def download_dataset(url, download_dir= "data/"):
    """Download dataset from Kaggle into the specified directory."""
    try:
        os.makedirs(download_dir, exist_ok=True)
        logger.info(f"📥 Starting dataset download from {url} ...")
        od.download(url, data_dir=download_dir)
        logger.info(f"✅ Dataset downloaded successfully to {download_dir}")
    except Exception as e:
        logger.error(f"❌ Failed to download dataset: {e}")
        raise e

def create_data_yaml(download_dir = "data/"):
    """Create YOLO-compatible data.yaml."""
    dataset_path = os.path.join(download_dir, "face-detection-dataset")
    data_yaml_path = os.path.join(dataset_path, "data.yaml")

    try:
        data_yaml = f"""
train: {os.path.abspath(dataset_path)}/images/train
val: {os.path.abspath(dataset_path)}/images/val

nc: 1
names: ['face']
"""
        with open(data_yaml_path, "w") as f:
            f.write(data_yaml.strip())

        logger.info(f"✅ data.yaml created at {data_yaml_path}")
    except Exception as e:
        logger.error(f"❌ Failed to create data.yaml: {e}")
        raise e

if __name__ == "__main__":
    DATASET_URL = "https://www.kaggle.com/datasets/fareselmenshawii/face-detection-dataset"

    # Step 1: Download the dataset
    download_dataset(DATASET_URL, download_dir="data/")

    # Step 2: Create YOLO data.yaml file
    create_data_yaml(download_dir="data/")
