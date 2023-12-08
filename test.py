# from Detection.components.data_ingestion import DataIngestion
# from Detection.components.data_validation import DataValidation
# from Detection.components.model_trainer import ModelTrainer
# import sys, os
# import yaml
import time
# from Detection.logger import logging
# from Detection.exception import AppException

# from Detection.entity.config_entity import (DataIngestionConfig,
#                                              DataValidationConfig,
#                                               ModelTrainerConfig )

# from Detection.entity.artifacts_entity import (DataIngestionArtifact,
#                                                 DataValidationArtifact,
#                                                   ModelTrainerArtifact)

class RuntimeTests:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_validation_config = DataValidationConfig()
        self.model_trainer_config = ModelTrainerConfig()


    
    def start_data_ingestion(self)-> DataIngestionArtifact:
        try: 
            logging.info(
                "Entered the start_data_ingestion method of TrainPipeline class"
            )
            logging.info("Getting the data from URL")

            data_ingestion = DataIngestion(
                data_ingestion_config =  self.data_ingestion_config
            )

            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logging.info("Got the data from URL")
            logging.info(
                "Exited the start_data_ingestion method of TrainPipeline class"
            )

            return data_ingestion_artifact

        except Exception as e:
            raise AppException(e, sys)
        


    
    def start_data_validation(
        self, data_ingestion_artifact: DataIngestionArtifact
    ) -> DataValidationArtifact:
        logging.info("Entered the start_data_validation method of TrainPipeline class")

        try:
            data_validation = DataValidation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_config=self.data_validation_config,
            )

            data_validation_artifact = data_validation.initiate_data_validation()

            logging.info("Performed the data validation operation")

            logging.info(
                "Exited the start_data_validation method of TrainPipeline class"
            )

            return data_validation_artifact

        except Exception as e:
            raise AppException(e, sys) from e
        
        

    def start_model_trainer(self
    ) -> ModelTrainerArtifact:
        try:
            model_trainer = ModelTrainer(
                model_trainer_config=self.model_trainer_config,
            )
            model_trainer_artifact = model_trainer.initiate_model_trainer()
            return model_trainer_artifact

        except Exception as e:
            raise AppException(e, sys)


# tester = RuntimeTests()
# data_ingestion_artifact_testrun = tester.start_data_ingestion()
# data_validation_artifact = tester.start_data_validation(data_ingestion_artifact_testrun)
# if data_validation_artifact.validation_status == True:
#     model_trainer_artifact = tester.start_model_trainer()


# updating yaml


def correct_paths_in_yaml(yaml_file_path):
    # Load the YAML file
    with open(yaml_file_path, 'r') as file:
        config = yaml.safe_load(file)

    # Update parameters
    config['test'] = 'D:\\projects\\DL\\AccidentResponseSystem\\artifacts\\data_ingestion\\test\\images'
    config['train'] = 'D:\\projects\\DL\\AccidentResponseSystem\\artifacts\\data_ingestion\\train\\images'
    config['val'] = 'D:\\projects\\DL\\AccidentResponseSystem\\artifacts\\data_ingestion\\valid\\images'

    # Write the updated content back to the file
    with open(yaml_file_path, 'w') as file:
        yaml.dump(config, file)

#correct_paths_in_yaml('testing.yaml')


start_time = time.time()

while True:
    # Capture a frame
    ret, frame = cap.read()

    if not ret:
        break  # Break the loop if there are no more frames

    # Check if it's time to make a prediction
    elapsed_time = time.time() - start_time
    if elapsed_time >= prediction_interval:
        # Perform object detection here using YOLOv8 or your chosen model

        # Reset the timer
        start_time = time.time()

    # Display the frame or do other processing as needed
    cv2.imshow('Frame', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break