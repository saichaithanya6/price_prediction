import logging
from abc import ABC, abstractmethod
import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

#Abstract Base Class for Data Splitting
class DataSplittingStrategy(ABC):
    @abstractmethod
    def split(self, data: pd.DataFrame, target: str, test_size: float = 0.2):
        '''Splits the data into training and testing sets'''
        pass

#Concrete class for Data Splitting
class SimpleTrainingSplit(DataSplittingStrategy):
    def __init__(self, test_size=0.2, random_state=42):
        
        self.test_size = test_size
        self.random_state = random_state
    
    def split(self, data: pd.DataFrame, target: str):
        '''Splits the data into training and testing sets'''
        logging.info("Starting data splitting")
        X = data.drop(columns=[target])
        y = data[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=42)
        logging.info("Data splitting completed")
        return X_train, X_test, y_train, y_test

#Context class for Data Splitting
class DataSplitter:
    def __init__(self, strategy: DataSplittingStrategy):
        '''Initializes the DataSplitter with a strategy'''
        self.strategy = strategy
    
    def set_strategy(self, strategy: DataSplittingStrategy):
        '''Sets the strategy of the DataSplitter'''
        self.strategy = strategy
    
    def execute_split(self, data: pd.DataFrame, target: str):
        '''Executes the strategy to split the data'''
        logging.info("Splitting data on selected strategy")
        
        return self.strategy.split(data, target)