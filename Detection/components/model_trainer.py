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
            
            os.system(f"yolo task=detect mode=train model=yolo8s.pt data=D:\projects\DL\AccidentResponseSystem\artifacts\data_ingestion\data.yaml epochs={self.model_trainer_config.no_epochs} imgsz=640")
            os.makedirs(self.model_trainer_config.model_trainer_dir, exist_ok=True)
            os.system(f"cp runs/detect/train/weights/best.pt {self.model_trainer_config.model_trainer_dir}/")
            os.system(f"cp runs/detect/train/confusion_matrix.png {self.model_trainer_config.model_trainer_dir}/")
            os.system(f"cp runs/detect/train/results.png {self.model_trainer_config.model_trainer_dir}/")
           
            # os.system("rm -rf runs")
            

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=f"{self.model_trainer_config.model_trainer_dir}/best.pt"
            )

            logging.info("initiate_model_trainer method of ModelTrainer class")
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")

            return model_trainer_artifact


        except Exception as e:
            raise AppException(e, sys)

