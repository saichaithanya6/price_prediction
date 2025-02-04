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
        print(data.info())

class summary_statistics_inspection(DataInspection):
    def inspect(self, data: pd.DataFrame):
        '''Inspects the summary statistics of the data'''
        print("In summary_statistics_inspection class")
        print("\n Summary Statistics of numerical features:")
        print(data.describe())
        print("\n Summary Statistics of categorical features:")
        print(data.describe(include = ['O']))


#This class is the context class that will use the DataInspection class
class DataInspector:
    def __init__(self, strategy: DataInspection):
        '''Initializes the DataInspector with a strategy'''
        print("In DataInspector class -- init method")
        self.strategy = strategy
    
    def set_strategy(self, strategy: DataInspection):
        '''Sets the strategy of the DataInspector'''
        print("In DataInspector class -- set_strategy method")
        self.strategy = strategy
    
    def execute_inspection(self, data: pd.DataFrame):
        '''Executes the strategy to inspect the data'''
        print("In DataInspector class -- execute_inspection method")
        self.strategy.inspect(data)

