import logging
import pandas as pd
from abc import ABC, abstractmethod
import numpy as np
from sklearn.base import RegressorMixin
from sklearn.metrics import mean_squared_error, r2_score


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Abstract Base Class for Model Evaluation

class ModelEvaluationStrategy(ABC):
    @abstractmethod
    def evaluate(self, model: RegressorMixin, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        '''Evaluates the model'''
        pass

#Concrete class for Model Evaluation
class RegressionModelEvaluation(ModelEvaluationStrategy):
    def evaluate(self, model: RegressorMixin, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        '''Evaluates the regression model'''
        logging.info("Evaluating regression model")
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        logging.info(f"Regression model evaluation completed  with mse: {mse}, r2: {r2}")
        return {"mse": mse, "r2": r2}

class ModelEvaluator:
    def __init__(self, strategy: ModelEvaluationStrategy):
        '''Initializes the ModelEvaluator with a strategy'''
        self.strategy = strategy
    
    def set_strategy(self, strategy: ModelEvaluationStrategy):
        '''Sets the strategy of the ModelEvaluator'''
        self.strategy = strategy
    
    def execute_evaluation(self, model: RegressorMixin, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        '''Executes the strategy to evaluate the model'''
        logging.info("Evaluating model on selected strategy")
        
        return self.strategy.evaluate(model, X_test, y_test)