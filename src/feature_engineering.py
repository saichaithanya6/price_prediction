import logging
from abc import ABC, abstractmethod

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler, StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Abstract Base Class for Feature Engineering

class FeatureEngineeringStrategy(ABC):
    @abstractmethod
    def transform(self, data: pd.DataFrame)-> pd.DataFrame:
        '''Transforms the data'''
        pass

#Concrete class for log transformation
class LogTransform(FeatureEngineeringStrategy):
    
    def __init__(self, features):
        self.features = features
    
    
    def apply_transform(self, data: pd.DataFrame)-> pd.DataFrame:
        '''Applies log transformation to the data'''
        logging.info("Applying log transformation")
        df_transformed = data.copy()
        for feature in self.features:
            df_transformed[feature] = np.log1p(df_transformed[feature]) # Using np.log1p to handle zero values
        logging.info("Log transformation completed")
        return df_transformed

#Concreate class for MinMaxScaler

class MinMaxScalerTransform(FeatureEngineeringStrategy):
    
    def __init__(self, features, feature_range=(0, 1)):
        self.features = features
        self.scaler = MinMaxScaler(feature_range=feature_range)
    
    def apply_transform(self, data: pd.DataFrame)-> pd.DataFrame:
        '''Applies Min-Max scaling to the data'''
        logging.info(f"Applying Min-Max scaling: {self.features} with range {self.scaler.feature_range}")
        df_transformed = data.copy()
        df_transformed[self.features] = self.scaler.fit_transform(df_transformed[self.features])
        logging.info("Min-Max scaling completed")
        return df_transformed

#Concrete class for StandardScaler
class StandardScalerTransform(FeatureEngineeringStrategy):
    
    def __init__(self, features):
        self.features = features
        self.scaler = StandardScaler()
    
    def apply_transform(self, data: pd.DataFrame)-> pd.DataFrame:
        '''Applies Standard scaling to the data'''
        logging.info(f"Applying Standard scaling: {self.features}")
        df_transformed = data.copy()
        df_transformed[self.features]= self.scaler.fit_transform(df_transformed[self.features])
        logging.info("Standard scaling completed")
        return df_transformed

#Concrete class for OneHotEncoder

class OneHotEncoderTransform(FeatureEngineeringStrategy):
    
    def __init__(self, features):
        self.features = features
        self.encoder = OneHotEncoder(sparse=False, drop='first')
    
    def apply_transform(self, data: pd.DataFrame)-> pd.DataFrame:
        '''Applies One-Hot encoding to the data'''
        logging.info(f"Applying One-Hot encoding: {self.features}")
        df_transformed = data.copy()
        encoded_df = pd.DataFrame(
            self.encoder.fit_transform(df_transformed[self.features]),
            columns= self.encoder.get_feature_names_out(self.features)
        )
        df_transformed= df_transformed.drop(columns= self.features)
        df_transformed = pd.concat([df_transformed, encoded_df], axis=1)
        logging.info("One-Hot encoding completed")
        return df_transformed

#Context class for Feature Engineering

class FeatureEngineer:
    def __init__(self, strategy: FeatureEngineeringStrategy):
        
        self.strategy = strategy
        
    
    def set_strategy(self, strategy: FeatureEngineeringStrategy):
        '''Sets the strategy of the FeatureEngineer'''
        logging.info("Setting new strategy")
        
        self.strategy = strategy
    
    def applying_feature_engineering(self, data: pd.DataFrame)-> pd.DataFrame:
        ''' Executes the strategy mentioned in the function '''
        logging.info("Applying feature engineering")
        
        return self.strategy.apply_transform(data)
        
    