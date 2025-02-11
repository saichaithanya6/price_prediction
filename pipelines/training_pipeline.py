from steps.data_ingestion_step import data_ingestion_step
from steps.feature_engineering_step import feature_engineering_step
from steps.handle_missing_values_step import handle_missing_values_step
from steps.model_building_step import model_building_step
from steps.model_evaluator_step import model_evaluator_step
from steps.data_splitter_step import data_splitter_step
from steps.outlier_detection_step import outlier_detection_step
from zenml import pipeline, Model, step

@pipeline(
    model= Model(name= "prices_predictor")
)

def ml_pipeline():
    """Defines end-to-end ML pipeline."""
    
    #Data ingestion
    raw_data = data_ingestion_step(file_path= "/data/archive.zip")
    
    #Handle missing values
    filled_data = handle_missing_values_step(df= raw_data)
    
    #Feature engineering
    transformed_data = feature_engineering_step(data= filled_data, strategy= "log", features= ["Gr Liv Area","SalePrice"])
    
    #Outlier detection
    cleaned_data = outlier_detection_step(data= transformed_data, column_name= "SalePrice")
    
    #Data splitting
    X_train, X_test, y_train, y_test = data_splitter_step(df= cleaned_data, target_column= "SalePrice")
    
    #Model Building Step
    model = model_building_step(X_train, y_train)
    
    #Model Evaluation Step
    evaluation_metrics, mse = model_evaluator_step(trained_model= model, X_test=X_test, y_test=y_test)
    
    return model

if __name__ == 'main':
    run= ml_pipeline()
    


