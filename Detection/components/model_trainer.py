import os,sys
import yaml
from Detection.utils.main_utils import read_yaml_file
from Detection.logger import logging
from Detection.exception import AppException
from Detection.entity.config_entity import ModelTrainerConfig
from Detection.entity.artifacts_entity import ModelTrainerArtifact



class ModelTrainer:
    def __init__(
        self,
        model_trainer_config: ModelTrainerConfig,
    ):
        self.model_trainer_config = model_trainer_config


    

    def initiate_model_trainer(self,) -> ModelTrainerArtifact:
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")

        try:
            logging.info("start training")
            
            #custom train the model
            os.system(f"yolo task=detect mode=train model=yolov8s.pt data=D:\\projects\\DL\\AccidentResponseSystem\\artifacts\\data_ingestion\\data.yaml epochs={self.model_trainer_config.no_epochs} imgsz=640")
            
            #create directory to store model trainer artifacts
            os.makedirs(self.model_trainer_config.model_trainer_dir, exist_ok=True)

            #paths for copying
            source_model = 'D:\\projects\\DL\AccidentResponseSystem\\runs\\detect\\train\weights\\best.pt'
            destination_model = f'{self.model_trainer_config.model_trainer_dir}\\'
            logging.info(f"source-model: {source_model} , destination-model: {destination_model}")
            source_matrix = 'D:\\projects\\DL\AccidentResponseSystem\\runs\\detect\\train\\confusion_matrix.png'
            destination_matrix = f'{self.model_trainer_config.model_trainer_dir}\\'
            logging.info(f"source-matrix: {source_matrix} , destination-matrix: {destination_matrix}")

            # Using the 'copy' command to copy the file
            logging.info('copying start')
            os.system(f'copy {source_model} {destination_model}')
            os.system(f'copy {source_matrix} {destination_matrix}')
            logging.info('copying complete')
            #Remove the runs folder created while training yolo
            # os.system("rm -rf runs")
            

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=f"{self.model_trainer_config.model_trainer_dir}\\best.pt"
            )

            logging.info("initiate_model_trainer method of ModelTrainer class")
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")

            return model_trainer_artifact


        except Exception as e:
            raise AppException(e, sys)

