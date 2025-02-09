import pandas as pd
from src.feature_engineering import FeatureEngineer, MinMaxScalerTransform, StandardScalerTransform, OneHotEncoderTransform, LogTransform
from zenml import step

@step
def feature_engineering_step(data: pd.DataFrame, strategy: str = "log", features: list = None)-> pd.DataFrame:
    if features is None:
        features = []
    '''Applies feature engineering to the data'''
    
    if strategy == "log":
        feature_engineer = FeatureEngineer(LogTransform(features))
    elif strategy == "minmax":
        feature_engineer = FeatureEngineer(MinMaxScalerTransform(features))
    elif strategy == "standard":
        feature_engineer = FeatureEngineer(StandardScalerTransform(features))
    elif strategy == "onehot":
        feature_engineer = FeatureEngineer(OneHotEncoderTransform(features))
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    transformed_data = feature_engineer.applying_feature_engineering(data)
    return transformed_data