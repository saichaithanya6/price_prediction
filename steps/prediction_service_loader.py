from zenml import step
from zenml.integrations.mlflow.model_deployers import MLFlowModelDeployer
from zenml.integrations.mlflow.services import MLFlowDeploymentService

@step(enable_cache=False)
def prediction_service_loader(pipeline_name: str, step_name: str) -> MLFlowDeploymentService:
    """
    Get the prediction service started by deployment pipeline
    """
    # get the Mlflow model deployer
    model_deployer = MLFlowModelDeployer.get_active_model_deployer()
    
    