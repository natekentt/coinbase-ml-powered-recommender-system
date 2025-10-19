import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, List
import time

# --- Path Configuration Fix ---
# Add project root to path to allow imports from the 'model' directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT) 

# Import necessary components from the model directory
# NOTE: We only import utilities. The main prediction logic is optimized for Streamlit caching here.
from model.train_model import preprocess_features, load_vocabularies, USER_FEATURE_NAMES, ITEM_FEATURE_NAMES, load_data_split

# --- File Paths ---
SAVED_MODELS_DIR = os.path.join(PROJECT_ROOT, 'saved_models')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data/mock_data')
USER_TOWER_PATH = os.path.join(SAVED_MODELS_DIR, 'user_tower.h5')
ASSET_TOWER_PATH = os.path.join(SAVED_MODELS_DIR, 'asset_tower.h5')
INVENTORY_SOURCE_PATH = os.path.join(DATA_DIR, 'test_data_split.pkl') 

# --- Constants for UI Inputs (Inferred from training data prep) ---
COUNTRY_OPTIONS = ['US', 'CA', 'GB', 'DE', 'AU', 'JP', 'IN', 'BR']
DEVICE_OPTIONS = ['desktop', 'mobile', 'tablet']
RISK_SCORE_RANGE = (1, 100)
DAYS_SINCE_TRADE_RANGE = (0, 365)
VIEWS_COUNT_RANGE = (0, 50)


# --- Initialization Functions (Cached for Performance) ---

@st.cache_resource
def load_and_prepare_resources():
    """
    Loads models, vocabularies, and prepares the static asset inventory (offline step).
    This function runs only once and caches the heavy objects.
    """
    try:
        # Suppress TensorFlow warnings
        tf.get_logger().setLevel('ERROR')
        
        # 1. Load Towers
        user_model = load_model(USER_TOWER_PATH, compile=False)
        asset_model = load_model(ASSET_TOWER_PATH, compile=False)
        
        # 2. Load Feature Vocabularies
        user_specs, item_specs = load_vocabularies()
        
        # 3. Prepare Asset Inventory (Offline Step Simulation)
        asset_features_raw_full, _ = load_data_split(INVENTORY_SOURCE_PATH)
        full_df = pd.DataFrame(asset_features_raw_full)
        item_df_raw = full_df[ITEM_FEATURE_NAMES]
        asset_df = item_df_raw.drop_duplicates(subset=['asset_ticker']).reset_index(drop=True)
        
        asset_features_raw = {name: asset_df[name].values for name in ITEM_FEATURE_NAMES}
        asset_features_processed = preprocess_features(asset_features_raw, item_specs) 
        
        # 4. Generate All Asset Embeddings (Cached)
        asset_embeddings = asset_model.predict(asset_features_processed, verbose=0)
        
        st.success(f"Models and inventory of {len(asset_df)} assets loaded successfully!")
        
        return user_model, asset_model, user_specs, item_specs, asset_df, asset_embeddings

    except Exception as e:
        st.error(f"Failed to load resources. Ensure all models and data files are present: {e}")
        st.stop()
        
def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """Calculates L2-normalized vectors (unit vectors) for distance calculation."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1e-6 
    return embeddings / norms

def run_prediction(
    user_model, user_specs, asset_df, asset_embeddings, user_features_dict: Dict, top_n: int = 10
):
    """
    Performs the fast prediction using cached components.
    """
    
    # 1. Process User Features (Real-Time Step)
    user_features_processed = preprocess_features(
        {k: np.array([v]) for k, v in user_features_dict.items()}, user_specs
    )

    # 2. Generate User Embedding
    user_embedding = user_model.predict(user_features_processed, verbose=0)
    user_embedding_normalized = normalize_embeddings(user_embedding)[0]

    # 3. Calculate Similarity and Rank (Candidate Generation)
    asset_embeddings_normalized = normalize_embeddings(asset_embeddings)
    similarity_scores = np.dot(asset_embeddings_normalized, user_embedding_normalized)

    # Get the indices of the top N scores
    top_indices = np.argsort(similarity_scores)[::-1][:top_n]
    
    # 4. Compile Results
    recommendations = []
    for rank, idx in enumerate(top_indices):
        recommendations.append({
            'Rank': rank + 1,
            'Ticker': asset_df.iloc[idx]['asset_ticker'],
            'Sector': asset_df.iloc[idx]['asset_sector'],
            'Market Cap Log': asset_df.iloc[idx]['market_cap_log'],
            'Score (Cosine)': similarity_scores[idx]
        })

    return pd.DataFrame(recommendations)

# --- Streamlit UI ---

# Load all heavy resources once
user_model, asset_model, user_specs, item_specs, asset_df, asset_embeddings = load_and_prepare_resources()


st.set_page_config(layout="wide", page_title="Coinbase Two-Tower Recommender Demo")

st.title("🚀 Two-Tower Crypto Recommender Demo")
st.markdown("Use the sidebar to define a user profile and see real-time asset recommendations generated by the trained two-tower model (AUC: **0.6323**).")

# --- Sidebar for User Inputs ---
with st.sidebar:
    st.header("👤 Define User Profile")
    
    # Input 1: Categorical
    user_country = st.selectbox(
        "User Country",
        options=COUNTRY_OPTIONS,
        index=0,
        help="Select the user's geographical location."
    )
    
    # Input 2: Categorical
    user_device_type = st.selectbox(
        "User Device Type",
        options=DEVICE_OPTIONS,
        index=0,
        help="Select the device used for viewing."
    )
    
    st.markdown("---")
    
    # Input 3: Numerical
    user_risk_score = st.slider(
        "User Risk Score (1-100)",
        min_value=RISK_SCORE_RANGE[0],
        max_value=RISK_SCORE_RANGE[1],
        value=20,
        step=5,
        help="Lower scores indicate more risk-averse behavior."
    )
    
    # Input 4: Numerical
    days_since_first_trade = st.slider(
        "Days Since First Trade",
        min_value=DAYS_SINCE_TRADE_RANGE[0],
        max_value=DAYS_SINCE_TRADE_RANGE[1],
        value=100,
        step=10,
        help="How long the user has been trading."
    )
    
    # Input 5: Numerical
    recent_views_count = st.slider(
        "Recent Views Count (30-day)",
        min_value=VIEWS_COUNT_RANGE[0],
        max_value=VIEWS_COUNT_RANGE[1],
        value=5,
        step=1,
        help="Number of different asset pages viewed recently."
    )
    
    st.markdown("---")
    top_n = st.slider("Number of Recommendations", 5, 25, 10, 1)

# --- Collect User Features ---
user_features = {
    'user_country': user_country,
    'user_device_type': user_device_type,
    'user_risk_score': user_risk_score,
    'days_since_first_trade': days_since_first_trade,
    'recent_views_count': recent_views_count
}

# --- Display User Profile (Improved Layout) ---
st.subheader("Current User Profile")
# Use columns and metrics to display the profile cleanly
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Risk Score", user_features['user_risk_score'])
with col2:
    st.metric("Days Active", user_features['days_since_first_trade'])
with col3:
    st.metric("Recent Views", user_features['recent_views_count'])
with col4:
    st.metric("Country", user_features['user_country'])
with col5:
    st.metric("Device", user_features['user_device_type'])


# --- Run Prediction Button and Display Results ---
st.markdown("---")
# Use a centered button to trigger the prediction
col_btn, _, _ = st.columns([1, 1, 1])

if col_btn.button("✨ Generate Recommendations", use_container_width=True, type="primary"):
    st.subheader(f"Top {top_n} Recommended Assets")
    
    # Use st.spinner for visual feedback
    with st.spinner("Generating real-time candidate assets..."):
        start_time = time.time()
        
        # Pass user_features_dict and cached resources to the predictor
        results_df = run_prediction(
            user_model, user_specs, asset_df, asset_embeddings, user_features, top_n
        )
        
        end_time = time.time()

    # Display the results
    st.dataframe(
        results_df, 
        hide_index=True, 
        use_container_width=True, 
        column_config={"Score (Cosine)": st.column_config.ProgressColumn("Score (Cosine)", format="%.4f", min_value=0, max_value=0.5)}
    )

    st.caption(f"Prediction time: {end_time - start_time:.4f} seconds.")
else:
    st.info("Adjust the user profile in the sidebar and click 'Generate Recommendations' to begin.")

st.markdown("""
<style>
    .stProgress > div > div > div > div {
        background-color: #004cff; /* Custom color for progress bar */
    }
</style>
""", unsafe_allow_html=True)
