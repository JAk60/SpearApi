import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt
import os

def plot_df_bar(df: DataFrame, mode =  "aggregate" , fig_path=None, ):
    """Plot a dataframe

    Args:
        df (DataFrame): a dataframe with numerical valued columns
        mode (str): Way to plot the bargraph
    """    
    if mode == "seperate":
        df.plot.bar(width=0.75,subplots=True,layout=(2,3),figsize=(15, 10))
    elif mode == "aggregate":
        df.plot.bar(width=0.75,subplots=True,layout=(2,3),figsize=(15, 10))
    
    if fig_path:
        # Check if the directory exists, and create it if it doesn't
        directory, fig_name = os.path.split(fig_path)
        
        os.makedirs(directory, exist_ok=True)

        fig_path = directory + "/" + fig_name 
        plt.savefig(fig_path)
        print(fig_path)
    
    # else:
    #     plt.show()
    
