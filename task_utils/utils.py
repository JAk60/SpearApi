import random

import numpy as np
import pandas as pd

import torch

import pandas as pd
# as the label for sub - mission is not available , so randomly assign label to sub-mission to complete the pipline
def assign_labels_for_sub_mission(df, sub_mission_labels=["Combat", "Exercise", "Sortie", "Humanitarian", "Fleet", "Support"]):
    condition1 = (df['Category'] == "Mission")
    condition2 = (df['Category'] == "Maintenance")

    # Create a copy of the DataFrame to avoid SettingWithCopyWarning
    df = df.copy()

    # Get the indices of rows that meet condition1
    indices = df.index[condition1]

    # Calculate the number of rows that should receive each value
    num_values = len(sub_mission_labels)
    num_rows = len(indices)
    rows_per_value = num_rows // num_values

    # Create a list of values to assign in the correct order
    assigned_values = (sub_mission_labels * (rows_per_value + 1))[:num_rows]

    # Assign the values to the 'Sub - mission' column using .loc
    df.loc[indices, 'Sub - mission'] = assigned_values

    # Assign "Not Applicable" to rows where condition2 is True using .loc
    df.loc[condition2, 'Sub - mission'] = "Not Applicable"

    # Convert 'Sub - mission' column to string type to ensure compatibility
    df['Sub - mission'] = df['Sub - mission'].astype(str)

    return df



def set_seed(seed):
    """
    Sets seed for all relevant libraries
    Args:
        seed (int): seed value for all modules
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
