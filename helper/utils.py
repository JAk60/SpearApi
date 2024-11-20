import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import os,sys
import pickle
from tqdm import tqdm

import sys
sys.path.append('../../')
sys.path.append('../')
sys.path.append('../codes/')

from sklearn.model_selection import train_test_split
from task_utils import embedding


def label_map(label_instances):
   return  {label: idx for idx, label in enumerate(label_instances)}




def divide_labeled_unlabeled(X_train, X_feats_train, Y_train, label_per, seed=42):
    # Set the random seed for reproducibility
    if seed is not None:
        np.random.seed(seed)
    
    # Determine the number of labeled samples
    num_samples = X_train.shape[0]
    num_labeled = int(label_per * num_samples)
    
    # Generate a random permutation of indices
    indices = np.random.permutation(num_samples)
    
    # Split indices into labeled and unlabeled
    labeled_indices = indices[:num_labeled]
    unlabeled_indices = indices[num_labeled:]
    
    # Create labeled and unlabeled subsets
    X_L = X_train[labeled_indices]
    X_feats_L = X_feats_train[labeled_indices]
    Y_L = Y_train[labeled_indices]
    
    X_U = X_train[unlabeled_indices]
    X_feats_U = X_feats_train[unlabeled_indices]
    
    return X_L, X_feats_L, Y_L, X_U, X_feats_U


def load_data_full(
        full_path="../../data/processed/version2/full.csv",
        embedding_path="../../data/processed/version2/full.npy",
        labels="Category", is_generate_embed = False):

    df_full = pd.read_csv(full_path)
    X = df_full["Scenario"]

    # if file embedding file is not exist please create the embedding
    if not os.path.exists(embedding_path) and is_generate_embed :
        features = embedding.scenarios_embedding(list(df_full["Scenario"]))
        np.save(embedding_path, features)

    with open(embedding_path, 'rb') as f:
        X_feats = np.load(f)

    label_instances = list(df_full[labels].value_counts().index)
    label_instances.sort()

    label_map_dict = label_map(label_instances)

    Y = df_full[labels]
    Y = df_full[labels].map(label_map_dict)
    Y = Y.to_numpy()
    
    return X, X_feats, Y, df_full

def load_data_train_test_split(
        processed_data_path="../../data/processed/",
        version=2,
        is_data_split=True, 
        labels="Category",
        is_generate_embed = False):
    
    if is_data_split:
        # Paths for training, validation, and test files
        train_path = f"{processed_data_path}version{version}/train.csv"
        train_embedding_path = f"{processed_data_path}version{version}/train.npy"
        val_path = f"{processed_data_path}version{version}/val.csv"
        val_embedding_path = f"{processed_data_path}version{version}/val.npy"
        test_path = f"{processed_data_path}version{version}/test.csv"
        test_embedding_path = f"{processed_data_path}version{version}/test.npy"
        
        # Load training data
        X_train, X_feats_train, Y_train, df_train = load_data_full(train_path, train_embedding_path, labels,is_generate_embed = is_generate_embed)
        
        # Load validation data
        X_val, X_feats_val, Y_val , df_val = load_data_full(val_path, val_embedding_path, labels,is_generate_embed = is_generate_embed)
        
        # Load test data
        X_test, X_feats_test, Y_test , df_test = load_data_full(test_path, test_embedding_path, labels, is_generate_embed = is_generate_embed)
        
        return X_train, X_feats_train, Y_train, X_val, X_feats_val, Y_val, X_test, X_feats_test, Y_test, df_train, df_val, df_test
    
    else:
        # Load the full data
        full_path = f"{processed_data_path}version{version}/full.csv"
        embedding_path = f"{processed_data_path}version{version}/full.npy"
        
        X, X_feats, Y , df_full = load_data_full(full_path, embedding_path, labels,is_generate_embed = is_generate_embed)
        
        return X, X_feats, Y, df_full



def get_various_data(X, Y, X_feats, val_per=0.15, test_per=0.15, label_per=0.1, U_per=None, seed=42):

    validation_size = int(val_per * len(X))
    test_size = int(test_per* len(X))
    train_size = len(X) - validation_size - test_size
    L_size =  int(label_per*train_size)
    if not U_per:
        U_size = train_size - L_size
    else :
        U_size = U_per * train_size

    if seed is not None:
        np.random.seed(seed)
    
    index = np.arange(X.size)
    index = np.random.permutation(index)
    X = X[index]
    Y = Y[index]
    X_feats = X_feats[index]

    X_V = X[-validation_size:]
    Y_V = Y[-validation_size:]
    X_feats_V = X_feats[-validation_size:]

    X_T = X[-(validation_size + test_size):-validation_size]
    Y_T = Y[-(validation_size + test_size):-validation_size]
    X_feats_T = X_feats[-(validation_size + test_size):-validation_size]

    X_L = X[-(validation_size + test_size + L_size):-(validation_size + test_size)]
    Y_L = Y[-(validation_size + test_size + L_size):-(validation_size + test_size)]
    X_feats_L = X_feats[-(validation_size + test_size + L_size):-(validation_size + test_size)]

    X_U = X[:U_size]
    X_feats_U = X_feats[:U_size]

    return X_V, Y_V, X_feats_V, X_T, Y_T, X_feats_T, X_L, Y_L, X_feats_L, X_U, X_feats_U


def get_test_U_data(X, Y, X_feats, test_per=0.30, U_per=None, random_seed=42):

    test_size = int(test_per* len(X))

    if random_seed is not None:
        np.random.seed(random_seed)
    
    if U_per is None:
        U_size = X.size - test_size
    else :
        U_size = U_per * X.size
    
    index = np.arange(X.size)
    index = np.random.permutation(index)
    X = X[index]
    Y = Y[index]
    X_feats = X_feats[index]

    X_T = X[-test_size:]
    Y_T = Y[-test_size:]
    X_feats_T = X_feats[-test_size:]

    X_U = X[:U_size]
    X_feats_U = X_feats[:U_size]

    return X_T, Y_T, X_feats_T, X_U, X_feats_U



def train_test_split_df(df, labels, test_size=0.15, val_size=0.15, random_state=42):

    val_test_size = test_size + val_size
    test_size = test_size / val_test_size
    
    df_train, df_test_val = train_test_split(df, test_size=val_test_size, random_state=42, stratify=df[labels])
    df_val, df_test = train_test_split(df_test_val, test_size=test_size, random_state=42, stratify=df_test_val[labels])

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




def print_all_shapes(X_V, Y_V, X_feats_V, X_T, Y_T, X_feats_T, X_L, Y_L, X_feats_L, X_U, X_feats_U):
    """
    Prints the shapes of the given variables in a formatted manner.
    
    Args:
        X_V, Y_V, X_feats_V, X_T, Y_T, X_feats_T, X_L, Y_L, X_feats_L, X_U, X_feats_U: Variables to print shapes of.
    """
    print(f'X_V shape: {X_V.shape}')
    print(f'Y_V shape: {Y_V.shape}')
    print(f'X_feats_V shape: {X_feats_V.shape}')
    print(f'X_T shape: {X_T.shape}')
    print(f'Y_T shape: {Y_T.shape}')
    print(f'X_feats_T shape: {X_feats_T.shape}')
    print(f'X_L shape: {X_L.shape}')
    print(f'Y_L shape: {Y_L.shape}')
    print(f'X_feats_L shape: {X_feats_L.shape}')
    print(f'X_U shape: {X_U.shape}')
    print(f'X_feats_U shape: {X_feats_U.shape}')


def process_data(is_data_split = True, model ="LR", processed_data_path = "../../data/processed/", version = 1, labels = "Category", test_per =0.15, val_per=0.15, label_per=None, seed=42, print_shape=False):
    if not is_data_split:
        X, X_feats, Y, df_full = load_data_train_test_split(
            processed_data_path=processed_data_path,
            version=version,
            is_data_split=is_data_split,
            labels=labels)
        
        if model == "CAGE":
            X_T, Y_T, X_feats_T, X_U, X_feats_U = get_test_U_data(X, Y, X_feats, test_per=test_per, U_per=None, random_seed=42)
            X_V, Y_V, X_feats_V = np.array([]), np.array([]), np.array([]) 
            X_L, Y_L, X_feats_L = np.array([]), np.array([]), np.array([]) 
            
        elif model == "JL":
            X_V, Y_V, X_feats_V, X_T, Y_T, X_feats_T, X_L, Y_L, X_feats_L, X_U, X_feats_U = get_various_data(X, Y, X_feats, val_per=val_per, test_per=test_per, label_per=label_per, U_per=None, seed=seed)
    else:
        X_train, X_feats_train, Y_train, X_V, X_feats_V, Y_V, X_T, X_feats_T, Y_T, df_train, df_val, df_test = load_data_train_test_split(
            processed_data_path=processed_data_path,
            version=version,
            is_data_split=is_data_split,
            labels=labels)
        
        if model == "CAGE":
            # Option to merge val with train
            X_U = np.hstack((X_train, X_V))
            Y_U = np.hstack((Y_train, Y_V))
            X_feats_U = np.vstack((X_feats_train, X_feats_V))
            X_V, Y_V, X_feats_V = np.array([]), np.array([]), np.array([]) 
            X_L, Y_L, X_feats_L = np.array([]), np.array([]), np.array([]) 
        elif model == "JL":
            X_L, X_feats_L, Y_L, X_U, X_feats_U = divide_labeled_unlabeled(X_train, X_feats_train, Y_train, label_per, seed)
    
    if print_shape:
        print_all_shapes(X_V, Y_V, X_feats_V, X_T, Y_T, X_feats_T, X_L, Y_L, X_feats_L, X_U, X_feats_U)

    return  X_V, X_feats_V, Y_V, X_T, X_feats_T, Y_T, X_L, Y_L, X_feats_L, X_U, X_feats_U

# Example usage
# X, X_feats, Y, df_full, X_train, X_feats_train, Y_train, X_V, X_feats_V, Y_V, X_T, X_feats_T, Y_T, df_train, df_val, df_test, X_L, Y_L, X_feats_L, X_U, Y_U, X_feats_U = process_data(is_data_split, model, processed_data_path, version, labels, test_per, val_per, label_per, seed, print_shape)