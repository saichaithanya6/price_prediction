import logging

import pandas as pd
from src.outlier_detection import OutlierDetector, ZScoreOutlierDetection, IQROutlierDetection

from zenml import step

@step
def outlier_detection_step(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """Detects outliers in the data using Z-Score method"""
    
    logging.info("Detecting outliers in the data")
    
    if df is None:
        raise ValueError("DataFrame is None")
    
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input is not a pandas DataFrame")
    
    if column_name not in df.columns:
        logging.error(f"Column {column_name} does not exist in the DataFrame")
        
    df_numeric= df.select_dtypes(include= [int, float])
    
    outlier_detector= OutlierDetector(ZScoreOutlierDetection(threshold= 3))
    
    outliers= outlier_detector.detect_outliers(df_numeric)
    df_cleaned= outlier_detector.handle_outliers(df_numeric, method= "remove")
    
    return df_cleaned