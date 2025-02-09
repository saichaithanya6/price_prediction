from typing import Tuple
import pandas as pd
from zenml import step
from src.data_splitter import DataSplitter, SimpleTrainingSplit

@step
def data_splitter_step(df: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split the data into training and testing sets."""
    splitter = DataSplitter(SimpleTrainingSplit())
    X_train, X_test, y_train, y_test = splitter.execute_split(df, target_column)
    return X_train, X_test, y_train, y_test


