import pandas as pd

from sklearn.model_selection import train_test_split

def train_test_split_df(df, labels):

    df_train, df_test_val = train_test_split(df, test_size=0.3, random_state=42, stratify=df[labels])
    df_val, df_test = train_test_split(df_test_val, test_size=0.5, random_state=42, stratify=df_test_val[labels])

    # Checking the resulting splits
    print(f"Training : {df_train.shape}, Validation : {df_val.shape}, Test : {df_test.shape}")

    print(f"Training : {df_train[labels].value_counts()}, Validation : {df_val[labels].value_counts()}, Test : {df_test[labels].value_counts()}")

    return df_train, df_val, df_test


def reverse_train_test_split(df_train, df_val, df_test):
    # Combine the validation and test DataFrames
    df_test_val = pd.concat([df_val, df_test])

    # Combine the training DataFrame with the combined validation and test DataFrames
    df = pd.concat([df_train, df_test_val])

    # Checking the resulting combined DataFrame
    print(f"Combined DataFrame : {df.shape}")

    return df