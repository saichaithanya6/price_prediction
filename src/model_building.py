import logging 
from abc import ABC, abstractmethod
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Abstract Base Class for Model Building
class ModelBuildingStrategy(ABC):
    @abstractmethod
    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> RegressorMixin:
        '''Builds the model'''
        pass

# Concrete class for Linear Regression Model Building
class LinearRegressionStrategy(ModelBuildingStrategy):    
    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
        '''Builds and trains the linear regression model'''
        
        #Ensure the inputs are correct
        
        if not isinstance(X_train, pd.DataFrame):
            raise ValueError("X_train must be a pandas DataFrame")
        if not isinstance(y_train, pd.Series):
            raise ValueError("y_train must be a pandas Series")
        
        logging.info("Initailizing Linear Regression Model")
        
        #Create a pipeline with standard scaler and linear regression
        
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression())
        ])
        logging.info("Pipeline created")
        pipeline.fit(X_train, y_train)
        logging.info("Model trained")
        return pipeline

class ModelBuilder:
    def __init__(self, strategy: ModelBuildingStrategy):
        '''Initializes the ModelBuilder with a strategy'''
        self.strategy = strategy
    
    def set_strategy(self, strategy: ModelBuildingStrategy):
        '''Sets the strategy of the ModelBuilder'''
        self.strategy = strategy
    
    def execute_build_and_train(self, X_train: pd.DataFrame, y_train: pd.Series) -> RegressorMixin:
        '''Executes the strategy to build and train the model'''
        logging.info("Building and training model on selected strategy")
        
        return self.strategy.build_and_train_model(X_train, y_train)
        
        
    


