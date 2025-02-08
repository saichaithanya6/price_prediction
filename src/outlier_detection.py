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