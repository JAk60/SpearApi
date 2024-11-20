import os
import numpy as np

import sys
sys.path.append('../../')
sys.path.append('../')
sys.path.append('../codes/')

from contextlib import redirect_stdout
from io import StringIO

from helper.accuracy import cage_accuracy_save_to_csv
from helper.create_dir_files_path import define_data_pipeline_paths, define_log_and_param_paths
from spear.cage import Cage
from spear.utils import get_data
import config
import warnings
warnings.filterwarnings('ignore')




def run_cage(
        version=1,
        labels="Category",
        model="CAGE",
        n_epochs=100,
        qt=0.9,
        qc=0.85,
        metric_avg=['micro', 'macro', 'weighted'],
        lr=0.01
    ):

    # Define paths
    (data_pipeline_path, full_pkl, full_path_json, path_json, 
     V_path_pkl, T_path_pkl, L_path_pkl, U_path_pkl) = define_data_pipeline_paths(version, labels, model)

    (log_path_1, params_path) = define_log_and_param_paths(version, labels, model)

    # Load data
    data_U = get_data(path=U_path_pkl, check_shapes=True)
    n_lfs = data_U[1].shape[1]

    # Initialize Cage
    cage = Cage(path_json=path_json, n_lfs=n_lfs)

    # Fit and predict probabilities
    with StringIO() as captured_output:
        with redirect_stdout(captured_output):
            probs = cage.fit_and_predict_proba(
                path_pkl=U_path_pkl,
                path_test=T_path_pkl,
                path_log=log_path_1,
                qt=qt,
                qc=qc,
                metric_avg=metric_avg,
                n_epochs=n_epochs,
                lr=lr
            )

        # Save parameters
        cage.save_params(params_path)

        # Extract and print captured output
        text = captured_output.getvalue().strip()
    print(text)

    # Save accuracy metrics
    accuracy_path = f"../../results/accuracy/version{version}/{model}/"
    os.makedirs(accuracy_path, exist_ok=True)
    cage_accuracy_save_to_csv(text, labels, accuracy_path)

if __name__ == "__main__":
    run_cage()
