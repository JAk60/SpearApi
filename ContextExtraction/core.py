from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import json
import asyncio
import traceback
import time
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('../')
from datetime import datetime
from typing import Dict, Any, Generator
from Qengine.Qengine import generate_classification_questions, filter_low_confidence_classifications
import math
import numpy as np
import traceback
from Qengine.Qengine import process_naval_classification
from i import pred_all_task

class Timer:
    def __init__(self):
        self.start_time = None
        self.checkpoints = {}

    def start(self):
        self.start_time = time.time()
        return self

    def checkpoint(self, name):
        self.checkpoints[name] = time.time() - self.start_time
        return self.checkpoints[name]

    def get_total(self):
        return time.time() - self.start_time

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        
        if isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return None
            return obj
        
        return super().default(obj)

def get_multi_label_classifications(probabilities: Dict[str, float], confidence_threshold: float = 0.02) -> Dict[str, float]:
    """
    Get all classifications above the confidence threshold for multi-label fields
    """
    return {class_name: prob for class_name, prob in probabilities.items() if prob > confidence_threshold}

def create_high_confidence_output(model_output: Dict[str, Dict[str, float]], refined_output: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """
    Creates high confidence output incorporating both model predictions and refined classifications
    """
    high_confidence_output = {}
    
    # Multi-label fields that should contain all classifications above threshold
    multi_label_fields = {'Task Objective', 'Constraints', 'Objective function'}
    
    # Process each layer from model output
    for layer_key, probabilities in model_output.items():
        if not probabilities:
            continue
            
        # Handle multi-label fields differently
        if layer_key in multi_label_fields:
            high_confidence_output[layer_key] = get_multi_label_classifications(probabilities)
        else:
            # For single-label fields, take the highest probability classification
            max_class = max(probabilities.items(), key=lambda x: x[1])
            high_confidence_output[layer_key] = {max_class[0]: max_class[1]}
    
    # Merge with refined output, giving priority to refined classifications
    for layer_key, refined_classes in refined_output.items():
        if refined_classes:  # Only update if refined classification exists
            if layer_key in multi_label_fields:
                # For multi-label fields, merge with existing classifications
                current_classes = high_confidence_output.get(layer_key, {})
                # Update with refined classifications, overwriting if same class exists
                current_classes.update(refined_classes)
                high_confidence_output[layer_key] = current_classes
            else:
                # For single-label fields, completely replace with refined classification
                high_confidence_output[layer_key] = refined_classes
    
    return high_confidence_output

def transform_predictions(pred_prob_all):
    """
    Transform prediction probabilities into the format expected by NavalClassifier
    """
    model_output = {
        'category': {},
        'sub category': {},
        'criticality': {},
        'Level': {},
        'Action': {},
        'Entity': {},
        'From': {},
        'Task Objective': {},
        'Constraints': {},
        'Objective function': {}
    }
    
    for key, probs in zip([
        "CATEGORY", "sub category", "criticality", "LEVEL", "Action", "Entity", 
        "From", "Task Objective", "Constraints", "Objective function",
    ], pred_prob_all):
        if key == "CATEGORY":
            model_output['category'] = probs
        elif key == "sub category":
            model_output['sub category'] = probs
        elif key == "LEVEL":
            model_output['Level'] = probs
        elif key in ["Hard Constraints", "Soft Constraints"]:
            model_output['Constraints'].update(probs)
        elif key in model_output:
            model_output[key] = probs
    
    return model_output

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.websocket("/predict")
async def predict(
    websocket: WebSocket,
    scenario: str = Query(...),
    version: int = Query(default=6),
    model: str = Query(default="BERT_classifier")
):
    timer = Timer().start()
    await websocket.accept()
    connection_time = timer.checkpoint("connection_accept")
    print(f"WebSocket connection established in {connection_time:.3f} seconds")
    
    qa_logs = []
    refined_output = {}  # Store the refined classifications
    
    try:
        # Get initial predictions
        pred_class_all, pred_prob_all = pred_all_task(
            scenario=scenario, 
            version=version, 
            model=model
        )
        
        # Transform predictions
        model_output = transform_predictions(pred_prob_all)
        print("--->>M/O",model_output)
        filtered_output = filter_low_confidence_classifications(model_output)
        
        # Process each classification layer
        for layer_key, probabilities in filtered_output.items():
            print(f"Processing layer: {layer_key}")
            refined_output[layer_key] = {}  # Initialize refined output for this layer
            
            # If there are probabilities for this layer, get the highest one
            if probabilities:
                max_class = max(probabilities.items(), key=lambda x: x[1])
                refined_output[layer_key][max_class[0]] = max_class[1]
                
            single_layer_input = {layer_key: probabilities}
            
            try:
                question_generator = generate_classification_questions(
                    single_layer_input, 
                    confidence_threshold=0.20
                )
                
                for question_data in question_generator:
                    # Send question to frontend
                    await websocket.send_text(json.dumps({
                        'question': question_data['question'],
                        'metadata': {
                            'layer_type': question_data['layer_type'],
                            'layer_key': question_data['layer_key'],
                            'class_1': question_data['class_1'],
                            'class_2': question_data['class_2'],
                            'prob_diff': question_data['prob_diff']
                        }
                    }))
                    
                    # Get answer as plain text
                    answer = await websocket.receive_text()
                    answer = answer.strip().lower()  # Clean the answer
                    
                    # Update refined output based on answer
                    if answer == 'yes':
                        if question_data['class_1'] in probabilities:
                            refined_output[layer_key] = {
                                question_data['class_1']: probabilities[question_data['class_1']]
                            }
                    elif answer == 'no':
                        if question_data['class_2'] and question_data['class_2'] in probabilities:
                            refined_output[layer_key] = {
                                question_data['class_2']: probabilities[question_data['class_2']]
                            }
                    
                    # Log Q&A
                    qa_log = {
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'layer_type': question_data['layer_type'],
                        'layer_key': question_data['layer_key'],
                        'question': question_data['question'],
                        'answer': answer,
                        'class_1': question_data['class_1'],
                        'class_2': question_data['class_2'],
                        'prob_1': question_data.get('prob_1', 0),
                        'prob_2': question_data.get('prob_2', 0),
                        'prob_diff': question_data['prob_diff'],
                        'is_multi_label': question_data.get('is_multi_label', False),
                        'refined_classification': refined_output[layer_key]
                    }
                    qa_logs.append(qa_log)
                
            except Exception as e:
                print(f"Error processing layer {layer_key}: {str(e)}")
                traceback.print_exc()
                continue
        
        # Create final high confidence output incorporating both sources
        high_confidence_output = create_high_confidence_output(model_output, refined_output)
        
        # Save logs
        if qa_logs:
            df = pd.DataFrame(qa_logs)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_filename = f'naval_classification_qa_logs_{timestamp}.csv'
            df.to_csv(log_filename, index=False)
        
        # Send final results
        total_time = timer.get_total()
        final_response = {
            'status': 'completed',
            'refined_classifications': refined_output,
            'high_confidence_classifications': high_confidence_output,
            'timing_summary': {
                'connection_time': connection_time,
                'total_time': total_time,
                'checkpoints': timer.checkpoints
            },
            'log_file': log_filename if qa_logs else None
        }
        
        await websocket.send_json(final_response)
        print("----->>>", final_response)
        
    except WebSocketDisconnect:
        print(f"Client disconnected after {timer.get_total():.3f} seconds")
    except Exception as e:
        error_time = timer.get_total()
        print(f"Error after {error_time:.3f} seconds:")
        traceback.print_exc()
        await websocket.send_text(json.dumps({
            "error": str(e),
            "error_time": error_time
        }))
    finally:
        if qa_logs and not any(log.get('log_file') for log in qa_logs):
            df = pd.DataFrame(qa_logs)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            final_log_filename = f'naval_classification_qa_logs_final_{timestamp}.csv'
            df.to_csv(final_log_filename, index=False)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)