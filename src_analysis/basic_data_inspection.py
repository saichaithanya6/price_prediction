from abc import ABC, abstractmethod
import pandas as pd


# Abstract Base Class for Data Inspection Strategies
# --------------------------------------------------
# This class defines a common interface for data inspection strategies.
class DataInspection(ABC):
    @abstractmethod
    def inspect(self, data: pd.DataFrame):
        '''Abstract method to inspect data'''
        pass

#This strategey inspects the data types of the columns

class DataTypeInspection(DataInspection):
    def inspect(self, data: pd.DataFrame):
        '''Inspects the data types of the columns'''
        return data.info()

class summary_statistics_inspection(DataInspection):
    def inspect(self, data: pd.DataFrame):
        '''Inspects the summary statistics of the data'''
        return data.describe()


#This class is the context class that will use the DataInspection class
class DataInspector:
    def __init__(self, strategy: DataInspection):
        '''Initializes the DataInspector with a strategy'''
        
        self.strategy = strategy
    
    def set_strategy(self, strategy: DataInspection):
        '''Sets the strategy of the DataInspector'''
        self.strategy = strategy
    
    def execute_inspection(self, data: pd.DataFrame):
        '''Executes the strategy to inspect the data'''
        return self.strategy.inspect(data)

