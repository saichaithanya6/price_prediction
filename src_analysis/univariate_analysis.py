from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Abstract Base Class for Univariate Analysis
# This class is used to define the structure of the univariate analysis

class UnivariateAnalysisStrategy(ABC):
    @abstractmethod
    def analyze(self, data: pd.DataFrame):
        '''Performs univariate analysis on the data'''
        pass
    

# Concrete class for numerical univariate analysis 

class NumericalUnivariateAnalysis(UnivariateAnalysisStrategy):
    def analyze(self, data:pd.DataFrame, feature: str):
        '''Performs univariate analysis on numerical data'''
        
        plt.figure(figsize=(12, 6))
        sns.histplot(data[feature], kde=True, bins=30, color='blue')
        plt.title(f'Distribution of {feature}')
        plt.xlabel(feature)
        plt.ylabel('Frequency')
        plt.show()

#Concrete class for categorical univariate analysis
class CategoricalUnivariateAnalysis(UnivariateAnalysisStrategy):
    def analyze(self, data:pd.DataFrame, feature: str):
        '''Performs univariate analysis on categorical data'''
        
        plt.figure(figsize=(15, 6))
        sns.countplot(x= feature, data=data, palette= 'muted')
        plt.title(f'Distribution of {feature}')
        plt.xlabel(feature)
        plt.xticks(rotation=45)
        plt.ylabel('Frequency')
        plt.show()

#Concrete class that uses univariate analysis

class UnivariateAnalysis:
    def __init__(self, strategy: UnivariateAnalysisStrategy):
        """
        Initializes the UnivariateAnalyzer with a specific analysis strategy.
        """
        self.strategy = strategy
    
    def set_strategy(self, strategy: UnivariateAnalysisStrategy):
        '''
        Sets the strategy for the univariate analysis'''
        self.strategy = strategy
        
    def execute_strategy(self, data: pd.DataFrame, feature: str):
        self.strategy.analyze(data, feature)
