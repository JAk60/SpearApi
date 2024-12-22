# %%
# %load_ext autoreload
# %autoreload 2

# %%
import sys
sys.path.append('../../')
sys.path.append('../')
sys.path.append('../codes/')

import os
import argparse
import numpy as np
import pandas as pd

from jl import run_jl
from cage import run_cage

from helper.utils import  get_various_data, get_test_U_data, load_data_train_test_split, divide_labeled_unlabeled, print_all_shapes, process_data

# %%

parser = argparse.ArgumentParser(description='train parameter')

parser.add_argument('--version', type=int, default=1, help='Version number')
parser.add_argument('--mode', type=str, choices=['deploy', 'exp', 'retrain'], default='exp', 
                    help='Mode of operation')
parser.add_argument('--label_per', type=float, default=0.2, 
                    help='Percentage of labels to consider (0.2 means 20 percent)')
parser.add_argument('--labels', type=str, default='Category', help='Label to consider')
parser.add_argument('--seed', type=int, default=42, help='Seed value')
parser.add_argument('--model', type=str, default='JL', help='Model name')
parser.add_argument('--task', type=str, default='all', help='Task to perform')
parser.add_argument('--val_per', type=float, default=0.15, help='Validation set percentage')
parser.add_argument('--test_per', type=float, default=0.15, help='Test set percentage')
parser.add_argument('--num_top_words', type=int, default=2, help='Number of top words')


parser.add_argument('--data_pipeline_available', choices=['True', 'False'], default='True', 
                    help='Whether data pipeline is available')
parser.add_argument('--is_data_split', choices=['True', 'False'], default='True', 
                    help='Whether data is split')
parser.add_argument('--full_data_lf', choices=['True', 'False'], default='False', 
                    help='Whether full data is loaded')


args = parser.parse_args()


# %%
# Define the variables
version = args.version
mode = args.mode
label_per = args.label_per
labels = args.labels
seed = args.seed
model = args.model
task = args.task
data_pipline_available = args.data_pipeline_available == "True"
is_data_split = args.is_data_split

val_per = args.val_per
test_per = args.test_per
num_top_words = args.num_top_words


print_shape = True
full_data_lf = False
is_generate_embed = False



# hyperparmeter tuning
loss_func_mask = [1, 1, 1, 1, 1, 1, 1]
batch_size = 32
lr_fm = 0.0005
lr_gm = 0.01
use_accuracy_score = True
feature_model = 'nn'
n_features = 768
n_hidden = 512
metric_avg = 'micro'
n_epochs = 100
start_len = 7
stop_len = 10
is_qt = True
is_qc = True
qt = 0.9
qc = 0.85


if model=="CAGE":
    metric_avg=['micro', 'macro', 'weighted']
lr=0.01




processed_data_path="../data/processed/"
full_path = processed_data_path+ "version" + str(version) + "/full.csv"
df_full = pd.read_csv(full_path)
all_tasks = df_full.columns[1:]

# %%
X_V, X_feats_V, Y_V, X_T, X_feats_T, Y_T, X_L, Y_L, X_feats_L, X_U, X_feats_U = process_data(
    is_data_split=is_data_split,
    model=model,
    processed_data_path=processed_data_path,
    version=version,
    labels=labels,
    test_per=test_per,
    val_per=val_per,
    label_per=label_per,
    seed=seed,
    print_shape=print_shape
)
# print("paths--->",X_V, X_feats_V, Y_V, X_T, X_feats_T, Y_T, X_L, Y_L, X_feats_L, X_U, X_feats_U )
# %%
print("data_pipline_available",data_pipline_available)
if data_pipline_available:
    print("data_pipline_available",data_pipline_available)

    from applyLF import run_applyLF
    
    if task == "all":
        for labels in all_tasks:
            label_instances = list(df_full[labels].value_counts().index)
            label_instances.sort()
            run_applyLF(
                    X_V, X_feats_V, Y_V, X_T, X_feats_T, Y_T, X_L, Y_L, X_feats_L, X_U, X_feats_U,
                    version=version,
                    labels=labels,
                    label_instances=label_instances,
                    model=model,
                    full_data_lf = full_data_lf,
                    seed=seed,
                    num_top_words = num_top_words
                )
        print("lf ran")
    else:
        label_instances = list(df_full[labels].value_counts().index)
        label_instances.sort()
        run_applyLF(
                X_V, X_feats_V, Y_V, X_T, X_feats_T, Y_T, X_L, Y_L, X_feats_L, X_U, X_feats_U,
                version=version,
                labels=labels,
                label_instances=label_instances,
                model=model,
                full_data_lf = full_data_lf,
                seed=seed,
                num_top_words = num_top_words
            )


# # %%
# if model == "JL":
#     if task == "all":
#         for labels in all_t asks:
#             label_instances = list(df_full[labels].value_counts().index)
#             label_instances.sort()
#             run_jl(
#                 version=version,
#                 labels=labels,
#                 model=model,
#                 loss_func_mask=loss_func_mask,
#                 batch_size=batch_size,
#                 lr_fm=lr_fm,
#                 lr_gm=lr_gm,
#                 use_accuracy_score=use_accuracy_score,
#                 feature_model=feature_model,
#                 n_features=n_features,
#                 n_hidden=n_hidden,
#                 metric_avg=metric_avg,
#                 n_epochs=n_epochs,
#                 start_len=start_len,
#                 stop_len=stop_len,
#                 is_qt=is_qt,
#                 is_qc=is_qc,
#                 qt=qt,
#                 qc=qc
#             )

#     else:
#             label_instances = list(df_full[labels].value_counts().index)
#             label_instances.sort()
#             # Function call to run_jl with the defined parameters
#             run_jl(
#                 version=version,
#                 labels=labels,
#                 model=model,
#                 loss_func_mask=loss_func_mask,
#                 batch_size=batch_size,
#                 lr_fm=lr_fm,
#                 lr_gm=lr_gm,
#                 use_accuracy_score=use_accuracy_score,
#                 feature_model=feature_model,
#                 n_features=n_features,
#                 n_hidden=n_hidden,
#                 metric_avg=metric_avg,
#                 n_epochs=n_epochs,
#                 start_len=start_len,
#                 stop_len=stop_len,
#                 is_qt=is_qt,
#                 is_qc=is_qc,
#                 qt=qt,
#                 qc=qc
#             )



# # %%
# if model == "CAGE":
#     if task == "all":
#         for labels in all_tasks:
#             label_instances = list(df_full[labels].value_counts().index)
#             label_instances.sort()
#             run_cage(
#                     version=version,
#                     labels=labels,
#                     model=model,
#                     n_epochs=n_epochs,
#                     qt=qt,
#                     qc=qc,
#                     metric_avg=metric_avg,
#                     lr=lr
#                 )

#     else:
#             label_instances = list(df_full[labels].value_counts().index)
#             label_instances.sort()
#             # Function call to run_jl with the defined parameters
#             run_cage(
#                     version=version,
#                     labels=labels,
#                     model=model,
#                     n_epochs=n_epochs,
#                     qt=qt,
#                     qc=qc,
#                     metric_avg=metric_avg,
#                     lr=lr
#                 )
# %%
