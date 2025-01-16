import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from transformers import DistilBertTokenizer, DistilBertModel
import torch

def print_dimensions(df):
    print("\nData Dimensions:")
    print(f"Total rows: {df.shape[0]}")
    print(f"Total columns: {df.shape[1]}")
    print("\nColumn types:")
    for dtype in df.dtypes.value_counts().index:
        cols = df.select_dtypes(include=[dtype]).columns
        print(f"{dtype}: {len(cols)} columns - {list(cols)}")

def split_and_create_embeddings(full_csv, text_columns=None, numeric_columns=None, train_size=0.7, val_size=0.15, random_state=42):
    print(f"Reading {full_csv}...")
    df = pd.read_csv(full_csv)
    
    print_dimensions(df)
    
    train_df, temp_df = train_test_split(df, train_size=train_size, random_state=random_state)
    relative_val_size = val_size / (1 - train_size)
    val_df, test_df = train_test_split(temp_df, train_size=relative_val_size, random_state=random_state)
    
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    
    train_df.to_csv('train.csv', index=False)
    val_df.to_csv('val.csv', index=False)
    test_df.to_csv('test.csv', index=False)
    print(f"\nSplit sizes:")
    print(f"Train: {len(train_df)} rows")
    print(f"Validation: {len(val_df)} rows")
    print(f"Test: {len(test_df)} rows")
    
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertModel.from_pretrained('distilbert-base-uncased')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    def encode_text(texts, batch_size=32):
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                all_embeddings.append(embeddings)
                
            print(f"Processed batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}", end='\r')
        
        return np.vstack(all_embeddings)
    
    def create_embeddings(df, output_filename):
        embeddings_list = []
        embedding_dimensions = 0
        
        if numeric_columns:
            numeric_cols = [col for col in numeric_columns if col in df.columns]
            if numeric_cols:
                scaler = StandardScaler()
                numeric_embeddings = scaler.fit_transform(df[numeric_cols])
                embeddings_list.append(numeric_embeddings)
                embedding_dimensions += numeric_embeddings.shape[1]
                print(f"\nNumeric features: {numeric_embeddings.shape[1]} dimensions")
        
        if text_columns:
            text_cols = [col for col in text_columns if col in df.columns]
            for col in text_cols:
                try:
                    texts = df[col].fillna('').astype(str).tolist()
                    col_embeddings = encode_text(texts)
                    embeddings_list.append(col_embeddings)
                    embedding_dimensions += col_embeddings.shape[1]
                    print(f"Text column '{col}': {col_embeddings.shape[1]} dimensions")
                except Exception as e:
                    print(f"Error processing column {col}: {str(e)}")
                    continue
        
        if not embeddings_list:
            raise ValueError("No embeddings were created. Check column specifications.")
            
        final_embeddings = np.hstack(embeddings_list) if len(embeddings_list) > 1 else embeddings_list[0]
        
        print(f"\nSaving to {output_filename}")
        print(f"Final embedding dimensions: {final_embeddings.shape[1]}")
        np.save(output_filename, final_embeddings)
        return final_embeddings
    
    print("\nProcessing full dataset...")
    create_embeddings(df, 'full.npy')
    
    print("\nProcessing training set...")
    create_embeddings(train_df, 'train.npy')
    
    print("\nProcessing validation set...")
    create_embeddings(val_df, 'val.npy')
    
    print("\nProcessing test set...")
    create_embeddings(test_df, 'test.npy')

if __name__ == "__main__":
    text_cols = ['Scenario']
    numeric_cols = None
    
    try:
        split_and_create_embeddings(
            './version6/full.csv',
            text_columns=text_cols,
            numeric_columns=numeric_cols
        )
    except Exception as e:
        print(f"An error occurred: {str(e)}")