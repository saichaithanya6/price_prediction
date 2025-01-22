import os
import zipfile
from abc import ABC, abstractmethod
import pandas as pd

# Abstract class for data ingestion
class DataIngestion(ABC):
    @abstractmethod
    def ingest(self, file_path) -> pd.DataFrame:
        '''Abstract method to ingest data from a file'''
        pass


#implement a concrete class for ingesting data from a zip file
class ZipFileDataIngestion(DataIngestion):
    def ingest(self, file_path: str) -> pd.DataFrame:
        '''Extracts a zip file and returns the content into a pandas DataFrame'''
        
        if not file_path.endswith(".zip"):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Extract the zip file
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall("extracted_data")
        
        # Find if there is a csv file in the extracted folder
        extracted_files = os.listdir("extracted_data")
        csv_files = [f for f in extracted_files if f.endswith(".csv")]
        
        if len(csv_files) == 0:
            raise FileNotFoundError(f"No CSV file found in the zip file: {file_path}")
        if len(csv_files) > 1:
            raise ValueError(f"Multiple CSV files found in the zip file: {file_path}")
        
        # Read the csv file into a pandas DataFrame
        csv_file_path = os.path.join("extracted_data", csv_files[0])
        df = pd.read_csv(csv_file_path)
        
        return df

class DataIngestorFactory:
    @staticmethod
    def get_data_ingestor(file_extension: str) -> DataIngestion:
        """Returns the appropriate DataIngestor based on file extension."""
        if file_extension == ".zip":
            return ZipFileDataIngestion()
        else:
            raise ValueError(f"No ingestor available for file extension: {file_extension}")



if __name__ == "__main__":
    # Example usage
    zip_file_path = "./data/archive.zip"
    file_extension = os.path.splitext(zip_file_path)[1]
    ingestion = DataIngestorFactory().get_data_ingestor(file_extension)
    df = ingestion.ingest(zip_file_path)
    print(df.head())
    
    pass