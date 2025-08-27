import opendatasets as od
import os
from src.utils.logger import logger  

class DataDownloader:
    def __init__(self, url, download_dir="data/"):
        self.url = url
        self.download_dir = download_dir

    def download(self):
        try:
            os.makedirs(self.download_dir, exist_ok=True)
            logger.info(f"📥 Starting download from {self.url} ...")
            od.download(self.url, data_dir=self.download_dir)
            logger.info(f"✅ Dataset downloaded to {self.download_dir}")
        except Exception as e:
            logger.error(f"❌ Failed to download dataset: {e}")
            raise e
        
