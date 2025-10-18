import pandas as pd
import numpy as np
from sklearn.preprocessing import QuantileTransformer
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from feature_store.offline_feature_loader import load_training_data

# --- Configuration ---
MAX_TOKENS = 10000  # Max vocabulary size for the Text Embedding
SEQUENCE_LENGTH = 32 # Max length of the asset description text sequence

# Define feature groups for clarity
USER_CATEGORICAL_FEATURES = ['region', 'risk_score', 'account_tier', 'favorite_sector']
ITEM_CATEGORICAL_FEATURES = ['ticker', 'sector']
ITEM_NUMERICAL_FEATURES = ['market_cap_usd', 'volatility_index']
ITEM_TEXT_FEATURE = 'asset_description_text'


def preprocess_numerical_features(df: pd.DataFrame, feature_name: str, transformer=None) -> tuple[pd.Series, QuantileTransformer]:
    """
    Applies a quantile transformation to numerical features to handle skewness
    and ensure uniform distribution for better model performance.
    
    In a real MLOps system, the fitted transformer would be saved/versioned
    and reused during serving to ensure consistency.
    """
    data = df[[feature_name]].copy()
    
    if transformer is None:
        # Fit the transformer for the first time (during training)
        transformer = QuantileTransformer(output_distribution='uniform', n_quantiles=len(data))
        transformed_data = transformer.fit_transform(data)
    else:
        # Reuse the fitted transformer (during serving/inference)
        transformed_data = transformer.transform(data)

    # Convert the numpy array back to a Series
    return pd.Series(transformed_data.flatten(), name=f'{feature_name}_norm'), transformer


def create_feature_pipeline(raw_df: pd.DataFrame) -> dict:
    """
    Orchestrates all feature engineering steps and organizes features 
    into separate dictionaries for the User and Item Towers.
    """
    
    # 1. Feature Dictionaries to be returned
    user_features = {}
    item_features = {}
    
    # --- 2. Numerical Feature Processing (Quantile Normalization) ---
    print("\n[Engineering] Processing Numerical Features...")
    
    # NOTE: In a production scenario, we would save the fitted transformers 
    # to disk and load them here. For this mock, we fit them during execution.
    
    for feature in ITEM_NUMERICAL_FEATURES:
        normalized_series, _ = preprocess_numerical_features(raw_df, feature)
        item_features[feature] = normalized_series.values
    
    # --- 3. Categorical Feature Processing (No transformation needed for pandas) ---
    # We will let the Keras preprocessing layers (StringLookup, CategoryEncoding)
    # handle the vocabulary and one-hot encoding *inside* the model definition.
    print("[Engineering] Categorical Features will be handled by Keras layers.")
    
    for feature in USER_CATEGORICAL_FEATURES:
        user_features[feature] = raw_df[feature].astype(str).values
        
    for feature in ITEM_CATEGORICAL_FEATURES:
        item_features[feature] = raw_df[feature].astype(str).values

    # --- 4. Text Feature Processing (Tokenization and Sequence Length) ---
    print("[Engineering] Preparing Text Feature for NLP Embedding.")
    # For now, we pass the raw text. The Keras TextVectorization layer will handle 
    # the heavy lifting (standardization, tokenization, vocab building) within the model.
    item_features[ITEM_TEXT_FEATURE] = raw_df[ITEM_TEXT_FEATURE].astype(str).values

    # --- 5. Target and Identifiers ---
    # User and Item IDs are required for the final train/test split, 
    # but not as model input features themselves.
    
    processed_output = {
        'user_features': user_features,
        'item_features': item_features,
        'target': raw_df['target'].values,
        'user_ids': raw_df['userId'].values,
        'asset_ids': raw_df['assetId'].values,
    }
    
    return processed_output


if __name__ == '__main__':
    # 1. Load the raw joined data
    raw_df = load_training_data()
    
    if not raw_df.empty:
        # 2. Run the feature engineering pipeline
        processed_data = create_feature_pipeline(raw_df)
        
        # 3. Print a summary of the processed output
        print("\n--- SUMMARY OF PROCESSED TENSOR INPUTS ---")
        
        print("\nUser Tower Inputs (Sample):")
        for k, v in processed_data['user_features'].items():
            print(f"  - {k}: dtype={v.dtype}, shape={v.shape}, sample='{v[0]}'")

        print("\nItem Tower Inputs (Sample):")
        for k, v in processed_data['item_features'].items():
            print(f"  - {k}: dtype={v.dtype}, shape={v.shape}, sample='{v[0]}'")
            
        print(f"\nTarget Shape: {processed_data['target'].shape}")
