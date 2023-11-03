import os.path
import sys
import yaml
import base64

from Detection.exception import AppException
from Detection.logger import logging


def read_yaml_file(file_path: str) -> dict:
    try:
        with open(file_path, "rb") as yaml_file:
            logging.info("Read yaml file successfully")
            return yaml.safe_load(yaml_file)

    except Exception as e:
        raise AppException(e, sys) from e
    



def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)

        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "w") as file:
            yaml.dump(content, file)
            logging.info("Successfully write_yaml_file")

    except Exception as e:
        raise AppException(e, sys)
    


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



def decodeImage(imgstring, fileName):
    imgdata = base64.b64decode(imgstring)
    with open("./data/" + fileName, 'wb') as f:
        f.write(imgdata)
        f.close()


def encodeImageIntoBase64(croppedImagePath):
    with open(croppedImagePath, "rb") as f:
        return base64.b64encode(f.read())

    
    