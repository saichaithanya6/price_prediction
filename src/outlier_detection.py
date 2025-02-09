import logging
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#Setup Logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Abstract Base Class for Outlier Detection
class OutlierDetection(ABC):
    @abstractmethod
    def detect_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        '''Detects outliers in the data'''
        pass

#Concrete class for Outlier Detection using Z-Score
class ZScoreOutlierDetection(OutlierDetection):
    def __init__(self, threshold=3):
        self.threshold = threshold
    
    def detect_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        '''Detects outliers using Z-Score method'''
        logging.info("Detecting outliers using Z-Score method")
        z_scores = np.abs((data - data.mean()) / data.std())
        outliers = z_scores > self.threshold
        logging.info(f"Outliers detected: {outliers.shape[0]}")
        return outliers
    
#Concrete class for Outlier Detection using IQR
class IQROutlierDetection(OutlierDetection):
    def detect_outliers(self, data):
        '''Detects outliers using IQR method'''
        logging.info("Detecting outliers using IQR method")
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = (data < lower_bound) | (data > upper_bound)
        logging.info(f"Outliers detected: {outliers.shape[0]}")
        return outliers
        

class OutlierDetector:
    def __init__(self, method: OutlierDetection):
        self.method = method
        
    def set_method(self, method: OutlierDetection):
        logging.info("Setting new outlier detection method")
        self.method = method
    
    def detect_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        '''Detects outliers using the selected method'''
        return self.method.detect_outliers(data)
    
    def handle_outliers(self, data: pd.DataFrame, method= "remove", **kwargs) -> pd.DataFrame:
        outliers = self.detect_outliers(data)
        if method == "remove":
            logging.info("Removing outliers")
            data_cleaned = data[(~outliers).all(axis=1)]
        elif method == "cap":
            logging.info("Capping outliers")
            data_cleaned = data.clip(lower= data.quantile(0.01), upper= data.quantile(0.99), axis= 1)
        else:
            logging.info("No action taken on outliers")
            data_cleaned = data
        return data_cleaned
    
    def visualize_outliers(self, data: pd.DataFrame, features: list):
        '''Visualizes outliers using boxplot'''
        for feature in features:
            plt.figure(fig_size=(12, 6))
            sns.boxplot(x=data[feature])
            plt.title(f"Boxplot of {feature}")
            plt.show()
        logging.info("Outliers visualized")
        
        