from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Abstract Base Class for Bivariate Analysis
class multivariate_analysis_template(ABC):

    def analyze(self, data: pd.DataFrame):
        '''Performs bivariate analysis on the data'''
        self.generate_correlation_heatmap(data)
        self.generate_pairplot(data)
        
    @abstractmethod
    def generate_correlation_heatmap(self, data: pd.DataFrame):
        '''Abstract method to generate correlation heatmap'''
        pass
    
    @abstractmethod
    def generate_pairplot(self, data: pd.DataFrame):
        '''Abstract method to generate pairplot'''
        pass
    
        
    
# Concrete class for numerical vs numerical multivariate analysis using corelation heatmap, pairplot

class Simplemultivariate_analysis(multivariate_analysis_template):
    
    def generate_correlation_heatmap(self, data: pd.DataFrame):
        '''Generates a correlation heatmap for numerical features'''
        plt.figure(figsize=(12, 10))
        sns.heatmap(data.corr(), annot=True, fmt ='.2f', cmap='coolwarm', linewidths=0.5)
        plt.title('Correlation Heatmap')
        plt.show()
    
    def generate_pairplot(self, data: pd.DataFrame):
        '''Generates a pairplot for numerical features'''
        sns.pairplot(data)
        plt.title('Pairplot')
        plt.show()
