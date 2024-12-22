import os

def create_directories(paths):
    """
    Create directories if they do not exist.

    Args:
        paths (list): List of directory paths to create.
    """
    for path in paths:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    print("Directories created or already exist.")

def define_data_pipeline_paths(version, labels, model):
    """
    Define data paths based on version, labels, and model.

    Args:
        version (str): Version of the data pipeline.
        labels (str): Labels for the data.
        model (str): Model name.

    Returns:
        tuple: Tuple containing data paths.
    """
    base_path = f"../checkpoint/data_pipeline/version{version}/{labels}/{model}/"
    data_pipeline_path = base_path
    full_pkl = os.path.join(base_path, f"{labels}_pickle.pkl")
    full_path_json = os.path.join(base_path, f"{labels}_full.json")
    path_json = os.path.join(base_path, f"{labels}.json")
    V_path_pkl = os.path.join(base_path, f"{labels}_pickle_V.pkl")
    T_path_pkl = os.path.join(base_path, f"{labels}_pickle_T.pkl")
    L_path_pkl = os.path.join(base_path, f"{labels}_pickle_L.pkl")
    U_path_pkl = os.path.join(base_path, f"{labels}_pickle_U.pkl")
    return (data_pipeline_path, full_pkl, full_path_json, path_json, 
            V_path_pkl, T_path_pkl, L_path_pkl, U_path_pkl)

def define_log_and_param_paths(version, labels, model):
    """
    Define log and parameter paths based on version, labels, and model.

    Args:
        version (str): Version of the data pipeline.
        labels (str): Labels for the data.
        model (str): Model name.

    Returns:
        tuple: Tuple containing log and parameter paths.
    """
    log_path_1 = f'../checkpoint/log/version{version}/{labels}/{model}/context_log_1.txt'
    params_path = f'../checkpoint/version{version}/{model}/{labels}.pkl'
    return (log_path_1, params_path)

def define_figure_paths(version, labels, model):
    """
    Define figure paths based on version, labels, and model.

    Args:
        version (str): Version of the data pipeline.
        labels (str): Labels for the data.
        model (str): Model name.

    Returns:
        tuple: Tuple containing figure paths.
    """
    fig_path = f"../../results/plots/version{version}/{labels}/{model}/"
    all_task_plot = f"../../results/plots/version{version}/all_task/{labels}.png"
    full_plot = os.path.join(fig_path, f"{labels}.png")
    V_path_plot = os.path.join(fig_path, f"{labels}_V.png")
    T_path_plot = os.path.join(fig_path, f"{labels}_T.png")
    L_path_plot = os.path.join(fig_path, f"{labels}_L.png")
    U_path_plot = os.path.join(fig_path, f"{labels}_U.png")
    return (fig_path, all_task_plot, full_plot, V_path_plot, T_path_plot, L_path_plot, U_path_plot)

def define_labeling_paths(version, labels, model):
    """
    Define labeling paths based on version, labels, and model.

    Args:
        version (str): Version of the data pipeline.
        labels (str): Labels for the data.
        model (str): Model name.

    Returns:
        tuple: Tuple containing labeling paths.
    """
    labeling_path = f"../../results/labeling/version{version}/{labels}/{model}/"
    full_labeling = os.path.join(labeling_path, f"{labels}.csv")
    V_path_labeling = os.path.join(labeling_path, f"{labels}_V.csv")
    T_path_labeling = os.path.join(labeling_path, f"{labels}_T.csv")
    L_path_labeling = os.path.join(labeling_path, f"{labels}_L.csv")
    U_path_labeling = os.path.join(labeling_path, f"{labels}_U.csv")
    return (labeling_path, full_labeling, V_path_labeling, T_path_labeling, L_path_labeling, U_path_labeling)

def run_create_dir_files_path(version, labels, model):
    """
    Run function to define paths and create directories.

    Args:
        version (str): Version of the data pipeline.
        labels (str): Labels for the data.
        model (str): Model name.

    Returns:
        tuple: Tuple containing all paths.
    """
    # Define paths
    (data_pipeline_path, full_pkl, full_path_json, path_json, 
     V_path_pkl, T_path_pkl, L_path_pkl, U_path_pkl) = define_data_pipeline_paths(version, labels, model)
    (log_path_jl_1, params_path) = define_log_and_param_paths(version, labels, model)
    (fig_path, all_task_plot, full_plot, V_path_plot, T_path_plot, L_path_plot, U_path_plot) = define_figure_paths(version, labels, model)
    (labeling_path, full_labeling, V_path_labeling, T_path_labeling, L_path_labeling, U_path_labeling) = define_labeling_paths(version, labels, model)
    
    # Collect all paths for directory creation
    all_paths = (data_pipeline_path, full_pkl, full_path_json, path_json, 
                 V_path_pkl, T_path_pkl, L_path_pkl, U_path_pkl,
                 log_path_jl_1, params_path,
                 fig_path, all_task_plot, full_plot, V_path_plot, T_path_plot, L_path_plot, U_path_plot,
                 labeling_path, full_labeling, V_path_labeling, T_path_labeling, L_path_labeling, U_path_labeling)
    
    # Create directories
    create_directories(all_paths)
    
    return all_paths

if __name__ == "__main__":
    version = "1"
    labels = "Category"
    model = "LR"

    all_paths = run_create_dir_files_path(version, labels, model)

    # Print all the paths
    for path in all_paths:
        print(path)
