import logging
import pandas as pd

import mlflow
from typing import Annotated
from zenml import step, Output
from sklearn.base import RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from zenml import ArtifactConfig, step
from zenml.client import Client

#Active experiment tracker
experiment_tracker = Client().active_experiment_tracker
from zenml import Model

model = Model(
    name= "prices_predictor",
    version= None,
    license= "Apache-2.0",
    description= "Pricepreediction for housing",
)

@step(enable_cache= True, experiment_tracker= experiment_tracker.name, model= model)
def model_building_step(
    X_train: pd.DataFrame, y_train: pd.Series
) -> Annotated[Pipeline, ArtifactConfig(name= "sklearn_pipeline", is_model_artifact= True)]:
    # Ensure the inputs are correct
    if not isinstance(X_train, pd.DataFrame):
        raise ValueError("X_train must be a pandas DataFrame")
    if not isinstance(y_train, pd.Series):
        raise ValueError("y_train must be a pandas Series")
    
    #identify numeric and categorical columns
    categorical_col= X_train.select_dtypes(include= ["object", "category"]).columns
    numerical_col= X_train.select_dtypes(exclude= ["object", "category"]).columns
    
    logging.info(f"Numerical columns: {numerical_col}")
    logging.info(f"Categorical columns: {categorical_col}")
    
    # Define the preprocessing steps for numerical and categorical columns
    numerical_transformer= SimpleImputer(strategy= "mean")
    categorical_transformers= Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy= "most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown= "ignore")),
        ]
    )
    
    #Bundle preprocessing for numerical and categorical columns
    preprocessor= ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_col),
            ("cat", categorical_transformers, categorical_col),
        ]
    )
    
    #Define the model traning pipeline
    pipeline= Pipeline(steps=[("preprocessor", preprocessor), ("model", LinearRegression())])
    
    #Start the MLflow run to log the model
    if not mlflow.active_run():
        mlflow.start_run()
    
    try:
        #Enable autologging for scikit-learn to automatically capture parameters, metrics, and model
        mlflow.sklearn.autolog()
        
        logging.info("Building and training the Linear Regression model...")
        # Fit the model
        pipeline.fit(X_train, y_train)
        logging.info("Model training completed.")
        
        #Log the column that the model expects
        onehot_encoder= (
            pipeline.named_steps["preprocessor"].transformers_[1][1].named_steps["onehot"]
        )
        onehot_encoder.fit(X_train[categorical_col])
        expected_columns= numerical_col.tolist() + list(onehot_encoder.get_feature_names_out(categorical_col))
        
        logging.info(f"Model expects the following columns: {expected_columns}")
    
    except Exception as e:
        logging.error(f"An error occurred during model training: {e}")
        raise e
    
    finally:
        # End the MLflow run
        mlflow.end_run()
    
    return pipeline
        