import logging
import pandas as pd
from abc import ABC, abstractmethod

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

#Abstract Base Class for Handling Missing Values

class MissingValuesHandler(ABC):
    @abstractmethod
    def handle(self, data: pd.DataFrame, strategy: str, columns=None):
        '''Handles missing values in the data'''
        pass
    
#Concrete class for Handling Missing Values
class DropMissingValuesStrategy(MissingValuesHandler):
    def __init__(self, axis= 0, thresh= None):
        
        self.axis = axis
        self.thresh = thresh
        
    def handle(self, df: pd.DataFrame)-> pd.DataFrame:
        
        logging.info("Dropping missing values with axis")
        
        df_cleaned = df.dropna(axis=self.axis, thresh=self.thresh)
        logging.info("Missing values dropped")
        return df_cleaned
    
#Concrete class for filling Missing Values
class FillMissingValuesStrategy(MissingValuesHandler):
    def __init__(self, method= "mean", fill_value=None):
        
        self.method = method
        self.fill_value = fill_value
        
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        
        logging.info(f"Filling missing values with {self.method}")
        df_cleaned = df.copy()
        
        if self.method == "mean":
            numeric_cols = df.cleaned.select_dtypes(include=["number"]).columns
            df_cleaned[numeric_cols]= df_cleaned[numeric_cols].fillna(df_cleaned[numeric_cols].mean())   
            
            
        elif self.method == "median":
            numeric_cols = df.cleaned.select_dtypes(include=["number"]).columns
            df_cleaned[numeric_cols]= df_cleaned[numeric_cols].fillna(df_cleaned[numeric_cols].median())
        
        elif self.method == "mode":
            nummeric_cols = df.cleaned.select_dtypes(include=["number"]).columns
            df_cleaned[numeric_cols]= df_cleaned[numeric_cols].fillna(df_cleaned[numeric_cols].mode().iloc[0], inplace=True) 
        
        elif self.method == "constant":
            df_cleaned = df.fillna(self.fill_value)
        
        else:
            raise ValueError(f"Unsupported method: {self.method}")
        logging.info("Missing values filled")
        
        return df_cleaned
    
#Context class for Handling Missing Values
class MissingValueHandler:
    def __init__(self, strategy: MissingValuesHandler):
        '''Initializes the MissingValueHandler with a strategy'''
        self.strategy = strategy
    
    def set_strategy(self, strategy: MissingValuesHandler):
        '''Sets the strategy of the MissingValueHandler'''
        self.strategy = strategy
    
    def handling_missing_values(self, data: pd.DataFrame):
        '''Executes the strategy to handle missing values'''
        logging.info("Handling missing values on selected strategy")
        
        return self.strategy.handle(data) 
        