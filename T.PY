import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from spear.jl.core import JL  
from task_utils import embedding
from spear.utils import get_data
# Step 1: Text Preprocessing
texts = ["An unidentified missile has been detected on radar, approaching the fleet at high speed. Its trajectory indicates a possible direct impact. All defense systems must be activated immediately to intercept the threat."]  # Your input text
data_U = get_data(path='./checkpoint/data_pipeline/version6/Objective function/JL/Objective function_pickle_U.pkl', check_shapes=True)
n_lfs = data_U[1].shape[1]
print("n_lfs",n_lfs)
# ------------>>> ../checkpoints/data_pipeline/version2/Action/JL/Action.json
# ------------>>> ../checkpoints/version2/JL/Action.pkl
scenario_embedding= embedding.scenarios_embedding(texts)
print(scenario_embedding.shape)
# Step 2: Initialize the JL Model
jl = JL(
    path_json="./checkpoint/data_pipeline/version6/Objective function/JL/Objective function.json",  # Path to your JSON file for class mapping
    n_lfs=n_lfs,  # Number of labeling functions
    n_features=4224,  # Number of features matches your text features
    feature_model='nn',  # Neural network as the feature model
    n_hidden=512  # Hidden layer size (if using 'nn')
)

# Step 3: Load Pretrained Model Parameters
jl.load_params("./checkpoint/version6/JL/Objective function.pkl")  

# Step 4: Predict Using the Feature Model
predictions = jl.predict_fm(scenario_embedding, need_strings=True)  
print("Predicted Classes:", predictions)

# Alternatively, get probabilities
probabilities = jl.predict_fm_proba(scenario_embedding)
print("Class Probabilities:", probabilities)
