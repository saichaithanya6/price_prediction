import json

import numpy as np
import pandas as pd
from zenml import step
from zenml.integrations.mlflow.services import MLFlowDeploymentService

@step(enable_cache=False)
def predictor(service: MLFlowDeploymentService, input_data: str) ->np.ndarray:
    #Start the prediction service
    service.start(timeout= 15)
    
    #Load the input data
    data = json.loads(input_data)
    
    #Extract the actual data and expected columns
    data.pop("columns", None)
    data.pop("index", None)
    
    #The expected columns
    expected_cols= [
        "Order",
        "PID",
        "MS SubClass",
        "Lot Frontage",
        "Lot Area",
        "Overall Qual",
        "Overall Cond",
        "Year Built",
        "Year Remod/Add",
        "Mas Vnr Area",
        "BsmtFin SF 1",
        "BsmtFin SF 2",
        "Bsmt Unf SF",
        "Total Bsmt SF",
        "1st Flr SF",
        "2nd Flr SF",
        "Low Qual Fin SF",
        "Gr Liv Area",
        "Bsmt Full Bath",
        "Bsmt Half Bath",
        "Full Bath",
        "Half Bath",
        "Bedroom AbvGr",
        "Kitchen AbvGr",
        "TotRms AbvGrd",
        "Fireplaces",
        "Garage Yr Blt",
        "Garage Cars",
        "Garage Area",
        "Wood Deck SF",
        "Open Porch SF",
        "Enclosed Porch",
        "3Ssn Porch",
        "Screen Porch",
        "Pool Area",
        "Misc Val",
        "Mo Sold",
        "Yr Sold",
    ]
    
    #convert the data into dataframe with correct columns
    df= pd.DataFrame(data["data"], columns=expected_cols)
    
    #Convert DF to JSON list for prediction
    json_list = json.loads(json.dumps(list(df.T.to_dict().values())))
    data_array = np.array(json_list)
    
    #Prediction
    prediction= service.predict(data_array)
    
    return prediction
    
    
    
    
    