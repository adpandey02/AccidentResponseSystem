from Detection.components.data_ingestion import DataIngestion



from Detection.entity.config_entity import (DataIngestionConfig,
                                                )

from Detection.entity.artifacts_entity import (DataIngestionArtifact,
                                                   )

data_ingestion = DataIngestion(
                data_ingestion_config =  DataIngestionConfig
            )

data_ingestion_artifact = data_ingestion.initiate_data_ingestion()

print(data_ingestion_artifact)