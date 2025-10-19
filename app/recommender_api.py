import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import sys # New: Required for path manipulation
import numpy as np
import pandas as pd
import pickle
from typing import Dict, List, Tuple

# --- Configuration & Path Fix ---

# 1. Calculate PROJECT_ROOT first (the directory above 'app')
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 2. FIX: Add the project root to sys.path so Python can find 'model.train_model'
# This resolves the ModuleNotFoundError when running the script from outside the root.
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT) 

# Note: The 'app/recommender_api.py' file must import utility functions 
# (load_vocabularies, preprocess_features, feature lists, load_data_split) 
# from the sibling directory 'model/train_model.py'. 
from model.train_model import preprocess_features, load_vocabularies, USER_FEATURE_NAMES, ITEM_FEATURE_NAMES, load_data_split

SAVED_MODELS_DIR = os.path.join(PROJECT_ROOT, 'saved_models')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data/mock_data')

USER_TOWER_PATH = os.path.join(SAVED_MODELS_DIR, 'user_tower.h5')
ASSET_TOWER_PATH = os.path.join(SAVED_MODELS_DIR, 'asset_tower.h5')

# The original ASSET_METADATA_PATH was missing. We now use the test split 
# as the source to dynamically extract the unique asset inventory for demonstration.
INVENTORY_SOURCE_PATH = os.path.join(DATA_DIR, 'test_data_split.pkl') 

# --- Helper Function ---

def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """Calculates L2-normalized vectors (unit vectors) for distance calculation."""
    # Ensure vectors are not zero before division
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    # Replace 0 norms with 1e-6 to avoid division by zero
    norms[norms == 0] = 1e-6 
    return embeddings / norms

# --- Main Recommendation Logic ---

def generate_recommendations(user_features_dict: Dict, top_n: int = 10):
    """
    Simulates a real-time recommendation request using the saved tower models.
    
    1. Loads necessary models and data.
    2. Generates user embedding.
    3. Generates all asset embeddings (the inventory).
    4. Calculates the cosine similarity score between the user and all assets.
    5. Returns the top N ranked assets.
    """
    print("\n--- Starting Real-Time Recommendation Generation ---")
    
    # Load feature specs (vocabularies)
    user_specs, item_specs = load_vocabularies()
    # all_specs is no longer needed, use user_specs and item_specs directly

    # 1. Load Towers
    try:
        user_model = load_model(USER_TOWER_PATH, compile=False)
        asset_model = load_model(ASSET_TOWER_PATH, compile=False)
        print("Towers loaded successfully.")
    except Exception as e:
        print(f"ERROR: Could not load tower models. {e}")
        return

    # 2. Prepare Asset Inventory (Offline Step Simulation)
    print(f"Loading asset inventory from {os.path.basename(INVENTORY_SOURCE_PATH)} to extract unique item features...")
    
    # Load the features from the data split (we ignore labels '_')
    asset_features_raw_full, _ = load_data_split(INVENTORY_SOURCE_PATH)

    # Convert the raw feature dictionary to a pandas DataFrame for easier unique extraction
    full_df = pd.DataFrame(asset_features_raw_full)
    
    # Extract only the columns related to the item features
    item_df_raw = full_df[ITEM_FEATURE_NAMES]
    
    # Get the unique assets based on the primary key (asset_ticker) to form the inventory
    asset_df = item_df_raw.drop_duplicates(subset=['asset_ticker']).reset_index(drop=True)

    # Now convert this unique asset DataFrame back to the required dictionary format for the model
    asset_features_raw = {name: asset_df[name].values for name in ITEM_FEATURE_NAMES}
    
    # Convert asset features for the asset tower
    # FIX: Use item_specs only, not the combined all_specs
    asset_features_processed = preprocess_features(asset_features_raw, item_specs) 
    
    # Generate all asset embeddings
    asset_embeddings = asset_model.predict(asset_features_processed, verbose=0)
    asset_embeddings_normalized = normalize_embeddings(asset_embeddings)
    
    print(f"Asset inventory size (unique assets): {len(asset_df)}")

    # 3. Process User Features (Real-Time Step)
    print("Processing user features to generate User Embedding...")
    
    # Convert user features for the user tower (need to wrap scalar values in arrays)
    user_features_processed = preprocess_features(
        {k: np.array([v]) for k, v in user_features_dict.items()}, user_specs # FIX: Use user_specs only
    )

    # Generate user embedding
    user_embedding = user_model.predict(user_features_processed, verbose=0)
    user_embedding_normalized = normalize_embeddings(user_embedding)[0]

    # 4. Calculate Similarity and Rank (Candidate Generation)
    print(f"Calculating similarity with {len(asset_df)} assets...")
    
    # Cosine similarity is the dot product of normalized vectors
    # This gives us a score between -1 and 1 for every asset
    similarity_scores = np.dot(asset_embeddings_normalized, user_embedding_normalized)

    # Get the indices of the top N scores
    top_indices = np.argsort(similarity_scores)[::-1][:top_n]
    
    # 5. Compile Results
    recommendations = []
    for rank, idx in enumerate(top_indices):
        recommendations.append({
            'rank': rank + 1,
            'ticker': asset_df.iloc[idx]['asset_ticker'],
            'sector': asset_df.iloc[idx]['asset_sector'],
            'score': similarity_scores[idx]
        })

    return recommendations

if __name__ == '__main__':
    # Suppress verbose TensorFlow warnings
    tf.get_logger().setLevel('ERROR')

    # Example: Define a new, active user in the US with a low risk score
    # Note: Keys MUST match USER_FEATURE_NAMES from train_model.py
    EXAMPLE_USER_FEATURES = {
        'user_country': 'US',
        'user_device_type': 'desktop',
        'user_risk_score': 20, # Low risk
        'days_since_first_trade': 100,
        'recent_views_count': 5
    }

    print("=============================================")
    print("         MOCK USER RECOMMENDATION")
    print("=============================================")
    print(f"User Profile: {EXAMPLE_USER_FEATURES}")
    
    # Run the recommendation generation
    top_recommendations = generate_recommendations(EXAMPLE_USER_FEATURES, top_n=10)

    print("\n=============================================")
    print(f"   Top {len(top_recommendations)} Recommended Assets")
    print("=============================================")
    
    # Display the results
    for rec in top_recommendations:
        print(f"Rank {rec['rank']:<2}: {rec['ticker']:<6} (Score: {rec['score']:.4f}) | Sector: {rec['sector']}")
        
    print("=============================================")
