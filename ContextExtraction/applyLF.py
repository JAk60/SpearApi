import os
import numpy as np
import pandas as pd

import sys
sys.path.append('../../')
sys.path.append('../')
sys.path.append('../codes/')


from spear.labeling import PreLabels

from lfs import manual_trigWords, design_lf

from helper.plots import plot_df_bar
from helper.utils import  get_various_data, get_test_U_data, load_data_train_test_split, divide_labeled_unlabeled, print_all_shapes
from helper.create_dir_files_path import run_create_dir_files_path

def run_applyLF(
                X_V, X_feats_V, Y_V, X_T, X_feats_T, Y_T, X_L, Y_L, X_feats_L, X_U, X_feats_U,
                version=1, 
                labels="Category", 
                label_instances=None, 
                model="JL", 
                full_data_lf=False, 
                seed=42, 
                num_top_words=2, 
                
            ):
    print("--->Running applyLF")
    

   
 
        
    # if print_shape and model == "JL":


    # ## Define the data pipeline path with version
    # data_pipeline_path = f"../../checkpoints/data_pipeline/version{version}/{labels}/{model}/"

    # # Define file paths with version
    # full_pkl = os.path.join(data_pipeline_path, f"{labels}_pickle.pkl")
    # full_path_json = os.path.join(data_pipeline_path, f"{labels}_full.json")
    # path_json = os.path.join(data_pipeline_path, f"{labels}.json")
    # V_path_pkl = os.path.join(data_pipeline_path, f"{labels}_pickle_V.pkl")  # Validation data
    # T_path_pkl = os.path.join(data_pipeline_path, f"{labels}_pickle_T.pkl")  # Test data
    # L_path_pkl = os.path.join(data_pipeline_path, f"{labels}_pickle_L.pkl")  # Labeled data
    # U_path_pkl = os.path.join(data_pipeline_path, f"{labels}_pickle_U.pkl")  # Unlabeled data

    # # Define log and parameter paths with version
    # log_path_1 = f'../../checkpoints/log/version{version}/{labels}/{model}/context_log_1.txt'
    # params_path = f'../../checkpoints/version{version}/{model}/{labels}.pkl'

    # # Create directories if they do not exist
    # os.makedirs(os.path.dirname(data_pipeline_path), exist_ok=True)
    # os.makedirs(os.path.dirname(log_path_1), exist_ok=True)
    # os.makedirs(os.path.dirname(params_path), exist_ok=True)

    # print("Directories created or already exist.")

    # # Define figure paths with version
    # fig_path = f"../../results/plots/version{version}/{labels}/{model}/"
    # os.makedirs(os.path.dirname(fig_path), exist_ok=True)

    # all_task_plot = f"../../results/plots/version{version}/all_task/{labels}.png"
    # os.makedirs(os.path.dirname(all_task_plot), exist_ok=True)

    # full_plot = os.path.join(fig_path, f"{labels}.png")
    # V_path_plot = os.path.join(fig_path, f"{labels}_V.png")  # Validation data - have true labels
    # T_path_plot = os.path.join(fig_path, f"{labels}_T.png")  # Test data - have true labels
    # L_path_plot = os.path.join(fig_path, f"{labels}_L.png")  # Labeled data - have true labels
    # U_path_plot = os.path.join(fig_path, f"{labels}_U.png")  # Unlabeled data - don't have true labels

    # labeling_path = f"../../results/labeling/version{version}/{labels}/{model}/"
    # full_labeling = os.path.join(labeling_path, f"{labels}.csv")
    # V_path_labeling = os.path.join(labeling_path, f"{labels}_V.csv")  # Validation data path
    # T_path_labeling = os.path.join(labeling_path, f"{labels}_T.csv")  # Test data path
    # L_path_labeling = os.path.join(labeling_path, f"{labels}_L.csv")  # Labeled data path
    # U_path_labeling = os.path.join(labeling_path, f"{labels}_U.csv")  # Unlabeled data path

    # os.makedirs(os.path.dirname(labeling_path), exist_ok=True)

    # replace all above in simple functions

    (data_pipeline_path, full_pkl, full_path_json, path_json, 
                 V_path_pkl, T_path_pkl, L_path_pkl, U_path_pkl,
                 log_path_1, params_path,
                 fig_path, all_task_plot, full_plot, V_path_plot, T_path_plot, L_path_plot, U_path_plot,
                 labeling_path, full_labeling, V_path_labeling, T_path_labeling, L_path_labeling, U_path_labeling) = run_create_dir_files_path(version, labels, model)
    
    if not label_instances:
        full_path = "../data/processed/" + "version"+str(version) + "/full.csv"
        df_full = pd.read_csv(full_path)
        label_instances = list(df_full[labels].value_counts().index)
        
    label_instances.sort()

    print(label_instances)

   
    ClassLabels, rules = design_lf(labels, label_instances, version = version, num_top_words = num_top_words)
    
    num_classes = len(label_instances)


    if full_data_lf:

        X, X_feats, Y, df_full = load_data_train_test_split(
        version=version,
        is_data_split=False,
        labels=labels)

        R = np.zeros((X.shape[0],len(rules.get_lfs())))
        context_noisy_labels = PreLabels(name=labels,
                                    data=X,
                                    gold_labels=Y,
                                    rules=rules,
                                    labels_enum=ClassLabels,
                                    num_classes=num_classes)
        L,S = context_noisy_labels.get_labels()


        context_noisy_labels.generate_pickle(full_pkl)
        context_noisy_labels.generate_json(full_path_json)

        analyse = context_noisy_labels.analyse_lfs(plot=True)
        analyse.to_csv(full_labeling, index =False)
        plot_df_bar(df=analyse, mode =  "aggregate" , fig_path = all_task_plot)
        plot_df_bar(df=analyse, mode =  "aggregate" , fig_path = full_plot)


    else:
        
        if model != "CAGE":

            context_noisy_labels = PreLabels(name=labels,
                                        data=X_V,
                                        gold_labels=Y_V,
                                        data_feats=X_feats_V,
                                        rules=rules,
                                        labels_enum=ClassLabels,
                                        num_classes=num_classes)
            context_noisy_labels.generate_pickle(V_path_pkl)
            context_noisy_labels.generate_json(path_json) #generating json files once is enough
            print("V_path_pkl",V_path_pkl)

            analyse = context_noisy_labels.analyse_lfs(plot=True)
            analyse.to_csv(V_path_labeling, index =False)
            plot_df_bar(df=analyse, mode =  "aggregate" , fig_path = V_path_plot)




            context_noisy_labels = PreLabels(name=labels,
                                        data=X_L,
                                        gold_labels=Y_L,
                                        data_feats=X_feats_L,
                                        rules=rules,
                                        labels_enum=ClassLabels,
                                        num_classes=num_classes)
            context_noisy_labels.generate_pickle(L_path_pkl)
            print("L_path_pkl",L_path_pkl)

            analyse = context_noisy_labels.analyse_lfs(plot=True)
            analyse.to_csv(L_path_labeling, index =False)
            plot_df_bar(df=analyse, mode =  "aggregate" , fig_path = L_path_plot)



        context_noisy_labels = PreLabels(name=labels,
                                    data=X_T,
                                    gold_labels=Y_T,
                                    data_feats=X_feats_T,
                                    rules=rules,
                                    labels_enum=ClassLabels,
                                    num_classes=num_classes)
        context_noisy_labels.generate_pickle(T_path_pkl)
        print("T_path_pkl",T_path_pkl)
        analyse = context_noisy_labels.analyse_lfs(plot=True)
        analyse.to_csv(T_path_labeling, index =False)
        plot_df_bar(df=analyse, mode =  "aggregate" , fig_path = T_path_plot)

        if model =="CAGE":
            context_noisy_labels.generate_json(path_json) 

        context_noisy_labels = PreLabels(name=labels,
                                    data=X_U,
                                    rules=rules,
                                    data_feats=X_feats_U,
                                    labels_enum=ClassLabels,
                                    num_classes=num_classes) # note that we don't pass gold_labels here, for the unlabelled data
        context_noisy_labels.generate_pickle(U_path_pkl)
        print("U_path_pkl",U_path_pkl)
        analyse = context_noisy_labels.analyse_lfs(plot=True)
        analyse.to_csv(U_path_labeling, index =False)
        plot_df_bar(df=analyse, mode =  "aggregate" , fig_path = U_path_plot)


if __name__ == "__main__":
    # Call the function with default parameters
    run_applyLF()
