import os
import numpy as np

import sys
sys.path.append('../../')
sys.path.append('../')
sys.path.append('../codes/')

from contextlib import redirect_stdout
from io import StringIO



# Assuming these imports are needed for the function
from spear.jl import JL
from spear.utils import get_data
from helper.accuracy import accuracy_save_to_csv, accuracy_per_save_to_csv, accuracy_round_save_to_csv
from helper.create_dir_files_path import define_data_pipeline_paths, define_log_and_param_paths
import config

import warnings
warnings.filterwarnings('ignore')

def run_jl(
        version=1,
        labels="Category",
        model="JL",
        loss_func_mask=[1, 1, 1, 1, 1, 1, 1],
        batch_size=32,
        lr_fm=0.0005,
        lr_gm=0.01,
        use_accuracy_score=True,
        feature_model='nn',
        n_features=768,
        n_hidden=512,
        metric_avg='weighted',
        n_epochs=100,
        start_len=7,
        stop_len=10,
        is_qt=True,
        is_qc=True,
        qt=0.9,
        qc=0.85,
    ):

    (data_pipeline_path, full_pkl, full_path_json, path_json,
     V_path_pkl, T_path_pkl, L_path_pkl, U_path_pkl) = define_data_pipeline_paths(version, labels, model)

    (log_path_1, params_path) = define_log_and_param_paths(version, labels, model)

    # Load data
    data_L = get_data(path=L_path_pkl, check_shapes=True)
    n_lfs = data_L[1].shape[1]

    # Initialize JL
    jl = JL(
        path_json=path_json,
        n_lfs=n_lfs,
        n_features=n_features,
        n_hidden=n_hidden,
        feature_model=feature_model
    )

    # Fit and predict probabilities
    with StringIO() as captured_output:
        with redirect_stdout(captured_output):
            probs_fm, probs_gm = jl.fit_and_predict_proba(
                path_L=L_path_pkl,
                path_U=U_path_pkl,
                path_V=V_path_pkl,
                path_T=T_path_pkl,
                loss_func_mask=loss_func_mask,
                batch_size=batch_size,
                lr_fm=lr_fm,
                lr_gm=lr_gm,
                use_accuracy_score=use_accuracy_score,
                path_log=log_path_1,
                return_gm=True,
                n_epochs=n_epochs,
                start_len=start_len,
                stop_len=stop_len,
                is_qt=is_qt,
                is_qc=is_qc,
                qt=qt,
                qc=qc,
                metric_avg=metric_avg
            )

        # Save parameters
        jl.save_params(params_path)

        text = captured_output.getvalue().strip()
        
    print(text)

    accuracy_path = f"../../results/accuracy/version{version}/{model}/"
    os.makedirs(accuracy_path, exist_ok=True)

    accuracy_save_to_csv(text, labels, accuracy_path)
    accuracy_per_save_to_csv(text, labels, accuracy_path)
    accuracy_round_save_to_csv(text, labels, accuracy_path)

if __name__ == "__main__":
    run_jl()