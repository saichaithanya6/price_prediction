import logging
from typing import Tuple

import pandas as pd
from sklearn.pipeline import Pipeline
from src.model_evaluation import ModelEvaluator, RegressionModelEvaluation
from zenml import step, Output

@step(enable_cache=False)

def model_evaluator_step(
    trained_model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> Tuple[dict, float]:
    """ Evaluates the trained model"""
    
    #Ensure the inputs are of the correct type
    
    if not isinstance(X_test, pd.DataFrame):
        raise ValueError("X_test must be a pandas DataFrame")
    if not isinstance(y_test, pd.Series):
        raise ValueError("y_test must be a pandas Series")
    
    logging.info("Applying the same preprocessing to the test data")
    
    #Apply preprocessing  and model prediction
    X_test_processed= trained_model.named_steps["preprocessor"].transform(X_test)
    
    #Initialize the evaluator with the regression strategy
    evaluator= ModelEvaluator(RegressionModelEvaluation())
    
    # Perform evaluation
    evaluation_metrics= evaluator.execute_evaluation(trained_model.named_steps["model"], X_test_processed, y_test)
    
    #Ensure that the evaluation metrics are returned as dictionary
    if not isinstance(evaluation_metrics, dict):
        raise ValueError("Evaluation metrics must be a dictionary")
    mse= evaluation_metrics.get("mse", None)
    return evaluation_metrics, mse
    
    
    
