import pandas as pd
import numpy as np
import tensorflow as tf
import os
import pickle
from typing import Dict, List, Tuple
from sklearn.model_selection import train_test_split

# --- Configuration ---
# NOTE: This assumes the script is run from the project root or the 'data' directory is correctly relative.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data/mock_data')
RAW_DATA_PATH = os.path.join(DATA_DIR, 'mock_coinbase_data.csv')
VOCAB_PATH = os.path.join(DATA_DIR, 'feature_vocabularies.pkl')
os.makedirs(DATA_DIR, exist_ok=True)

# Define output paths for splits
TRAIN_DATA_PATH = os.path.join(DATA_DIR, 'train_data_split.pkl')
VAL_DATA_PATH = os.path.join(DATA_DIR, 'val_data_split.pkl')
TEST_DATA_PATH = os.path.join(DATA_DIR, 'test_data_split.pkl')

# Feature Definitions (Must match the columns generated in generate_mock_data.py)
USER_FEATURES = [
    'user_country', 
    'user_device_type', 
    'user_risk_score', 
    'days_since_first_trade', 
    'recent_views_count'
]
ITEM_FEATURES = [
    'asset_ticker', 
    'asset_sector', 
    'market_cap_log', 
    'asset_volatility'
]
LABEL_FEATURE = 'target_interaction' # Binary label (e.g., 1 for click/purchase, 0 otherwise)


def create_feature_vocabularies(df: pd.DataFrame) -> Tuple[Dict, Dict]:
    """
    Creates vocabularies for all categorical features and saves them.
    """
    user_specs = {}
    item_specs = {}
    
    # 1. Categorical Vocabularies
    for feature in ['user_country', 'user_device_type']:
        # Use unique values + 1 for OOV (Out-Of-Vocabulary) token
        vocab = df[feature].astype(str).unique().tolist()
        user_specs[feature] = {'vocab': vocab, 'is_categorical': True}

    for feature in ['asset_ticker', 'asset_sector']:
        vocab = df[feature].astype(str).unique().tolist()
        item_specs[feature] = {'vocab': vocab, 'is_categorical': True}

    # 2. Numerical Features (No vocab, but define shape: 1 for the Input Layer)
    for feature in ['user_risk_score', 'days_since_first_trade', 'recent_views_count']:
        user_specs[feature] = {'is_categorical': False, 'shape': 1}

    for feature in ['market_cap_log', 'asset_volatility']:
        item_specs[feature] = {'is_categorical': False, 'shape': 1}
        
    # Save the specifications
    vocabs = {'user_specs': user_specs, 'item_specs': item_specs}
    with open(VOCAB_PATH, 'wb') as f:
        pickle.dump(vocabs, f)
    
    print(f"Saved feature vocabularies to {VOCAB_PATH}")
    return user_specs, item_specs

def transform_and_split_data(df: pd.DataFrame):
    """
    Splits the DataFrame into train, val, and test sets using index splitting,
    and transforms results into NumPy arrays for saving.
    """
    
    # 1. Get the indices for the entire dataframe and the labels for stratification
    indices = df.index.values 
    labels = df[LABEL_FEATURE].values.astype(np.float32)

    # 2. Split indices for train (80%) and temp (20%)
    # NOTE: We split the INDICES, but use the LABELS for stratification
    train_indices, temp_indices, _, _ = train_test_split(
        indices, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # 3. Split temp into validation (50% of temp) and test (50% of temp)
    # We must slice the labels array based on temp_indices for stratification in the second split
    temp_labels = labels[temp_indices]
    val_indices, test_indices, _, _ = train_test_split(
        temp_indices, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
    )

    # 4. Repack and save
    data_splits = [
        ('train', train_indices, TRAIN_DATA_PATH),
        ('val', val_indices, VAL_DATA_PATH),
        ('test', test_indices, TEST_DATA_PATH),
    ]

    all_features = USER_FEATURES + ITEM_FEATURES
    
    for name, indices, path in data_splits:
        # Slice the entire dataframe using the split indices
        df_split = df.iloc[indices]
        
        # Extract features and labels into NumPy arrays
        split_features = {f: df_split[f].values for f in all_features}
        split_labels = df_split[LABEL_FEATURE].values.astype(np.float32)

        data_to_save = {
            'features': split_features,
            'labels': split_labels
        }
        with open(path, 'wb') as f:
            pickle.dump(data_to_save, f)
        print(f"Saved {name} split data to {path} (Samples: {len(indices)})")


# --- Main Execution ---
if __name__ == '__main__':
    print("--- Starting Feature Engineering Pipeline ---")
    
    if not os.path.exists(RAW_DATA_PATH):
        print(f"Error: Raw data file not found at {RAW_DATA_PATH}. Please ensure 'mock_coinbase_data.csv' exists.")
    else:
        # Load data
        df = pd.read_csv(RAW_DATA_PATH)
        
        # 1. Create Vocabularies
        create_feature_vocabularies(df)
        
        # 2. Transform and Split Data
        # Re-index the dataframe before splitting to ensure contiguous indices for slicing
        df_indexed = df.reset_index(drop=True) 
        transform_and_split_data(df_indexed)
        
        print("\n--- Feature Engineering Complete. Ready for Training! ---")
