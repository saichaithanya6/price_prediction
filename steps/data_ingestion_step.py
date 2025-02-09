import pandas as pd
from src.ingest_data import DataIngestorFactory
from zenml import step

@step
def data_ingestion_step(file_path: str):
    '''Ingest data from a zip file'''
    #Determine file extension
    file_extension = ".zip"
    data_ingestor = DataIngestorFactory.get_data_ingestor(file_extension)
    
    #Ingest data and load it into a pandas DataFrame
    data= data_ingestor.ingest(file_path)
    return data