import os

from pipelines.training_pipeline import ml_pipeline
from zenml import pipeline
from steps.dynamic_importer import dynamic_importer
from steps.model_loader import model_loader
from steps.prediction_service_loader import prediction_service_loader
from steps.predictor import predictor
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step

#Define the requirements file for the pipeline
file= os.path.join(os.path.dirname(__file__), "requirements.txt")

@pipeline
def continous_deployment_pipeline():
    #Run the training pipeline
    trained_model = ml_pipeline()
    
    #Deploy the trained model
    mlflow_model_deployer_step(workers = 3, deploy_decision= True, model= trained_model)
    

@pipeline
def inference_pipeline():
    '''Run the batch inference job with data loaded from API
    '''
    #Load the batch data
    batch_data= dynamic_importer()
    
    #Load the deployed model service
    
    model_deployment_service= prediction_service_loader(
        pipeline_name= "continous_deployment_pipeline", step_name= "mlflow_model_deployer_step"
    )
    
    #Make predictions
    predictor(service= model_deployment_service, input_data= batch_data)
        
    