import math
import json
import numpy as np
import traceback
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from i import pred_all_task

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictionRequest(BaseModel):
    scenario: str
    version: int = 2
    model: str = "BERT_classifier"

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        # Handle numpy types
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        
        # Handle numpy arrays
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        
        # Handle special float values
        if isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return None
            return obj
        
        return super().default(obj)

def safe_serialize(data):
    """
    Safely serialize data, handling numpy and special float values
    """
    return json.loads(json.dumps(data, cls=CustomJSONEncoder))

@app.post("/predict")
async def predict(request: PredictionRequest):
    try:
        print(f"Prediction request: {request}")
        
        # Call prediction function
        pred_class_all, pred_prob_all = pred_all_task(
            scenario=request.scenario, 
            version=request.version, 
            model=request.model
        )
        
        # Keys for mapping probabilities
        keys = [
            "CATEGORY", "MissionType","LEVEL", "Action", "Entity", "From", "Time",
            "Location", "Task Objective", "Objective function", "Hard Constraints", "Soft Constraints"
        ]
        
        # Prepare the formatted response
        formatted_response = []

        for key, pred_probs in zip(keys, pred_prob_all):
            items = []
            for category, value in pred_probs.items():  # assuming pred_probs is a dictionary of categories and probabilities
                items.append({"category": category, "value": value})
            
            formatted_response.append({
                "name": key,
                "items": items
            })
        
        # Use custom JSON encoder to prepare the response
        return Response(
            content=json.dumps(formatted_response, cls=CustomJSONEncoder),
            media_type="application/json"
        )
    
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Prediction error: {error_details}")
        
        raise HTTPException(
            status_code=500, 
            detail=f"Prediction error: {str(e)}"
        )
