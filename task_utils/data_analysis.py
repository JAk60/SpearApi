import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def assign_labels_for_sub_mission(df, sub_mission_labels=["Combat", "Exercise", "Sortie", "Humanitarian", "Fleet", "Support"]):
    condition1 = (df['Category'] == "Mission")
    condition2 = (df['Category'] == "Maintenance")

    values_to_assign = sub_mission_labels

    # Create a copy of the DataFrame to avoid SettingWithCopyWarning
    df = df.copy()

    # Get the indices of rows that meet the condition
    indices = df[condition1].index

    # Calculate the number of rows that should receive each value
    num_values = len(values_to_assign)
    num_rows = len(indices)
    rows_per_value = num_rows // num_values

    # Create a list of values to assign in the correct order
    assigned_values = (values_to_assign * (rows_per_value + 1))[:num_rows]

    # Assign the values to the rows using .loc
    df.loc[indices, 'Sub - mission'] = assigned_values

    # Assign "Not Applicable" to rows where condition2 is True using .loc
    df.loc[condition2, 'Sub - mission'] = "Not Applicable"

    return df


def create_count_plots(columns_to_plot, df):
    # List of columns to plot

    if isinstance(columns_to_plot,str):
        columns_to_plot = [columns_to_plot]

    fig_type = len(columns_to_plot)
    # Set up the matplotlib figure with subplots

    n_rows = int(np.ceil(len(columns_to_plot)/3))

    if fig_type == 1:
        fig, axs = plt.subplots(n_rows, 1, figsize=(10, 6 * n_rows))
        axs = [axs]
    else:
        fig, axs = plt.subplots(n_rows, 3, figsize=(24, 6 * n_rows))
        axs = axs.flatten()[:fig_type]

    # Create a bar plot for each column using seaborn
    for idx, column in enumerate(columns_to_plot):
        sns.countplot(x=column, data=df, palette='viridis', hue=column, dodge=False, legend=False, ax=axs[idx])
        
        # Add titles and labels
        axs[idx].set_title(f'Distribution of {column}', fontsize=16)

        axs[idx].set_xlabel(column, fontsize=14)
        axs[idx].set_ylabel('Count', fontsize=14)
        
        # Add count labels on top of the bars
        level_counts = df[column].value_counts()

        for i in range(len(level_counts)):
            
            axs[idx].text(i, level_counts.values[i] + 1, str(level_counts.values[i]), ha='center', fontsize=12)
        
    
        # axs[idx].tick_params('x',direction='out', length=6, width=2, colors='r',
        #        grid_color='r', grid_alpha=0.5)
        
        # labels = list(map(add_newline, list(level_counts.index)))
        # axs[idx].set_xticklabels(labels, ha='right', fontsize=10)


        # Improve the layout
        plt.tight_layout()

    # Show the plot
    plt.show()