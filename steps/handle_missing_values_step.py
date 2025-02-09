import pandas as pd
from src.handling_missing_values import MissingValueHandler, DropMissingValuesStrategy, FillMissingValuesStrategy

from zenml import step

@step
def handle_missing_values_step(df: pd.DataFrame, strategy: str= 'meean') -> pd.DataFrame:
    """Handles missing values in the data."""
    
    if strategy == "drop":
        handler = MissingValueHandler(DropMissingValuesStrategy(axis = 0))
    
    elif strategy in ["mean", "median", "mode", "constant"]:
        handler = MissingValueHandler(FillMissingValuesStrategy(strategy))
    else:
        raise ValueError(f"Invalid strategy: {strategy}")
    
    cleaned_df = handler.handling_missing_values(df)
    
    return cleaned_df
    