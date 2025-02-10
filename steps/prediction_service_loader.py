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
    
    # fetch existing services with same pipeline and step name
    existing_services = model_deployer.find_model_server(
        pipeline_name=pipeline_name, step_name=step_name
    )
    
    if not existing_services:
        raise ValueError(f"No active model server found for pipeline '{pipeline_name}' and step '{step_name}'.")
    
    return existing_services[0]
    