# %%
import os
import numpy as np
import pandas as pd
import json
import argparse


import sys
sys.path.append('../../')
sys.path.append('../')
sys.path.append('../codes/')

from helper.create_dir_files_path import define_data_pipeline_paths, define_log_and_param_paths

from spear.jl import JL
from spear.cage import Cage
from spear.utils import get_data

from spear.labeling import PreLabels, LFSet

from task_utils import embedding
# from lfs import design_lf

import BERT_classifier_inference

import warnings
warnings.filterwarnings('ignore')

# %%

# %%
def pred_task(scenario, version = 1, model ="JL", labels = "Category",scenario_embedding = None):
    if scenario_embedding is None :
        scenario_embedding= embedding.scenarios_embedding(scenario)

    (data_pipeline_path, full_pkl, full_path_json, path_json, 
        V_path_pkl, T_path_pkl, L_path_pkl, U_path_pkl) = define_data_pipeline_paths(version, labels, model)


    (log_path_1, params_path) =  define_log_and_param_paths(version, labels, model)

        # Load data
    data_U = get_data(path=U_path_pkl, check_shapes=True)
    n_lfs = data_U[1].shape[1]
    # Initialize JL
    # Configuration settings

    feature_model = 'nn'
    n_features = 768
    n_hidden = 512

    if model =="JL":

        jl = JL(
            path_json=path_json,
            n_lfs=n_lfs,
            n_features=n_features,
            n_hidden=n_hidden,
            feature_model=feature_model
        )

        jl.load_params(params_path)
      

        pred_prob = jl.predict_fm_proba(scenario_embedding)
        pred = jl.predict_fm(scenario_embedding)
    
    elif model =="CAGE":

        from lfs import design_lf
        
        cage = Cage(path_json=path_json, n_lfs=n_lfs)
        cage.load_params(params_path)

        # in case of case instead of scenario, pickle path is required
        full_path = "../data/processed/version" + str(version) + "/full.csv"
        df_full = pd.read_csv(full_path)

        label_instances = list(df_full[labels].value_counts().index)
        label_instances.sort()
        ClassLabels, rules = design_lf(labels, label_instances, version = version, num_top_words = 2)

        context_noisy_labels = PreLabels(name=labels,
                            data=np.array([scenario]),
                            data_feats=np.array([scenario_embedding]),
                            rules = rules,
                            labels_enum=ClassLabels,
                            num_classes=len(label_instances ))
        
        temp_test_path = f"../../checkpoints/temp/test.pkl"
        os.makedirs(os.path.dirname(temp_test_path), exist_ok=True)
        context_noisy_labels.generate_pickle(temp_test_path)


        pred_prob = cage.predict_proba(temp_test_path) # change here
        pred_prob[0]= ["{:.16f}".format((p / sum(pred_prob[0]))) for p in pred_prob[0]]
        pred = cage.predict(temp_test_path)
    

    classes = json.load(open(path_json))
    pred_class = classes[str(pred[0])]

    pred_prob_dict = {classes[i] : pred_prob[0][int(i)] for i in classes }


    return pred_class, pred_prob_dict


# %%


# %%
def pred_all_task(scenario, version = 1, model ="JL",scenario_embedding = None):

    if scenario_embedding is None :
        scenario_embedding= embedding.scenarios_embedding(scenario)
    
    if model =="BERT_classifier":
        return BERT_classifier_inference.pred_all_task(scenario=scenario, version=version, scenario_embedding=scenario_embedding)
    

    pred_class_all = []
    pred_prob_all = []

    full_path = "../data/processed/version" + str(version) + "/full.csv"
    df_full = pd.read_csv(full_path)
    all_tasks = df_full.columns[1:]

    for labels in all_tasks:
        pred_class, pred_prob= pred_task(scenario, version = version, model = model , labels = labels , scenario_embedding = scenario_embedding)

        pred_class_all.append(pred_class)
        pred_prob_all.append(pred_prob)
    return pred_class_all, pred_prob_all


# %%
if __name__ == "__main__":



    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', type=str, required=True,
                        help='Enter Scenario to extract the context')
    parser.add_argument('--version', type=int, default=2,
                        help='Version number (default: 2)')
    parser.add_argument('--model', type=str, default='BERT_classifier',
                        help='Model name (default: BERT_classifier)')
    
    args = parser.parse_args()
    
    # Now you can access the arguments using args.scenario, args.version, and args.model
    scenario = args.scenario
    version = args.version
    model = args.model

    #     # Print or use the arguments as needed
    # print(f"Scenario: {scenario}")
    # print(f"Version: {version}")
    # print(f"Model: {model}")
    

    # pred = pred_task(scenario,labels = "Category", version = version, model =model)
    # print(pred)

    pred_class_all , pred_prob_all  = pred_all_task(scenario, version = version, model =model)
    print(pred_class_all , pred_prob_all )

   
# %%
