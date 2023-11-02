import os
import sys
import zipfile
from roboflow import Roboflow
import gdown
from Detection.logger import logging
from Detection.exception import AppException
from Detection.entity.config_entity import DataIngestionConfig
from Detection.entity.artifacts_entity import DataIngestionArtifact


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig = DataIngestionConfig()):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
           raise AppException(e, sys)
        

        
    def download_data(self)-> str:
        '''
        Fetch data from the url
        '''

        try: 
            
            download_dir = self.data_ingestion_config.data_ingestion_dir
            os.makedirs(download_dir, exist_ok=True)
            rf = Roboflow(api_key="0nnQNfnUUo68sGl8ozgr")
            project = rf.workspace("accident-detection-model").project("accident-detection-model")

            logging.info(f"Downloading data from Roboflow into file {download_dir}")
            dataset = project.version(2).download("yolov8", location=download_dir)

            logging.info(f"Downloaded data from Roboflow into file {download_dir}")

            return download_dir

        except Exception as e:
            raise AppException(e, sys)
        

    
    # def extract_zip_file(self,zip_file_path: str)-> str:
    #     """
    #     zip_file_path: str
    #     Extracts the zip file into the data directory
    #     Function returns None
    #     """
    #     try:
    #         feature_store_path = self.data_ingestion_config.feature_store_file_path
    #         os.makedirs(feature_store_path, exist_ok=True)
    #         with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    #             zip_ref.extractall(feature_store_path)
    #         logging.info(f"Extracting zip file: {zip_file_path} into dir: {feature_store_path}")

    #         return feature_store_path

    #     except Exception as e:
    #         raise AppException(e, sys)
        


    
    def initiate_data_ingestion(self)-> DataIngestionArtifact:
        logging.info("Entered initiate_data_ingestion method of Data_Ingestion class")
        try: 
            download_file_path = self.download_data()
            #feature_store_path = self.extract_zip_file(zip_file_path)

            data_ingestion_artifact = DataIngestionArtifact(
                download_file_path = download_file_path
                #feature_store_path = feature_store_path
            )

            logging.info("Exited initiate_data_ingestion method of Data_Ingestion class")
            logging.info(f"Data ingestion artifact: {data_ingestion_artifact}")

            return data_ingestion_artifact

        except Exception as e:
            raise AppException(e, sys)
        

    
    
        