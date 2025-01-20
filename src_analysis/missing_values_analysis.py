import os
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# Abstract Base Class for Missing Values Analysis

#This class defines a template for missing values analysis
class MissingValuesAnalysis(ABC):
    
    def analyze(self, data: pd.DataFrame):
        '''Performs missing values analysis on the data'''
        
        self.idetifying_missing_values(data)
        self.visualizing_missing_values(data)
    
    @abstractmethod
    def identifying_missing_values(self, data: pd.DataFrame):
        '''Abstract method to identify missing values in the data'''
        pass
    
    @abstractmethod
    def visualizing_missing_values(self, data: pd.DataFrame):
        '''Abstract method to visualize missing values in the data'''
        pass

#Concrete class for missing values analysis identification
#This class identifies and visualizes missing values in the data

class SimpleMissingValuesAnalysis(MissingValuesAnalysis):
    
    def identifying_missing_values(self, data: pd.DataFrame):
        '''Identifies missing values in the data'''
        missing_values = data.isnull().sum()
        print(missing_values)
    
    def visualizing_missing_values(self, data: pd.DataFrame):
        '''Visualizes missing values in the data using a heatmap'''
        plt.figure(figsize=(10, 5))
        sns.heatmap(data.isnull(), cbar=False, cmap='viridis')
        plt.title('Missing Values in the Data')
        plt.show()
        