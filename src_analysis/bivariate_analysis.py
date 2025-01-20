from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Abstract Base Class for Bivariate Analysis
# This defines a common interface for bivariate analysis

class BivariateAnalysis(ABC):
    @abstractmethod
    def analyze(self, data: pd.DataFrame, x: str, y: str):
        '''Performs bivariate analysis on the data'''
        pass

# Concrete class for numerical vs numerical bivariate analysis

class NumericalNumericalAnalysis(BivariateAnalysis):
    
    def analyze(self, data: pd.DataFrame, feature1: str, feature2: str):
        '''Performs numerical vs numerical bivariate analysis'''
        plt.figure(figsize=(10, 5))
        sns.scatterplot(data=data, x=feature1, y=feature2)
        plt.title(f'{feature1} vs {feature2}')
        plt.show()

# Concrete class for numerical vs categorical bivariate analysis
class NumericalCategoricalAnalysis(BivariateAnalysis):
    
    def analyze(self, data: pd.DataFrame, feature1: str, feature2: str):
        '''Performs numerical vs categorical bivariate analysis'''
        plt.figure(figsize=(10, 5))
        sns.boxplot(data=data, x=feature1, y=feature2)
        plt.title(f'{feature1} vs {feature2}')
        plt.xticks(rotation=45)
        plt.show()

# Context class for bivariate analysis strategy

class BivariateAnalysisContext:
    
    def __init__(self, strategy: BivariateAnalysis):
        '''Initializes the context with a strategy'''
        self.strategy = strategy
    
    def set_strategy(self, strategy: BivariateAnalysis):
        '''Sets the strategy for the context'''
        self.strategy = strategy
    
    def execute_analysis(self, data: pd.DataFrame, feature1: str, feature2: str):
        '''Performs bivariate analysis using the strategy'''
        self.strategy.analyze(data, feature1, feature2)

