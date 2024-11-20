import os
import numpy as np
import pandas as pd

import sys
sys.path.append('../../')
sys.path.append('../')
sys.path.append('../codes/')


import pickle

from task_utils import embedding

import warnings
warnings.filterwarnings('ignore')



def pred_task(scenario, version = 1, model ="BERT_classifier", labels = "Category",scenario_embedding = None, label_instances = None):

    params_path = f"../checkpoints/version{version}/BERT_classifier/"

    if scenario_embedding is None :
        scenario_embedding= embedding.scenarios_embedding(scenario)
    file_path = params_path+labels+".pkl"
    with open(file_path, 'rb') as f:
        model_2 = pickle.load(f)
    pred_class = model_2.predict(scenario_embedding)
    pred_prob = model_2.predict_proba(scenario_embedding)



    train_path = "../data/processed/" + "version"+str(version) + "/train.csv"
    df_full = pd.read_csv(train_path)

    label_instances = list(df_full[labels].value_counts().index)
    label_instances.sort()
    pred_prob_dict = {label_instances[i] : pred_prob[0][int(i)] for i in range(len(label_instances)) }
    
    return pred_class[0], pred_prob_dict


def pred_all_task(scenario, version = 1, model ="BERT_classifier",scenario_embedding = None):

    if scenario_embedding is None :
        scenario_embedding= embedding.scenarios_embedding(scenario)

    train_path = "../data/processed/" + "version"+str(version) + "/train.csv"
    df_full = pd.read_csv(train_path)


    pred_class_all = []
    pred_prob_all = []

    all_tasks = df_full.columns[1:]

    for labels in all_tasks:
        pred_class, pred_prob= pred_task(scenario, version = version, model = model , labels = labels , scenario_embedding = scenario_embedding)

        pred_class_all.append(pred_class)
        pred_prob_all.append(pred_prob)
    return pred_class_all, pred_prob_all