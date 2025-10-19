import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Embedding, Concatenate, Flatten
from tensorflow.keras.models import Model
import os
import pickle
from typing import Dict, List, Tuple

# --- Configuration (Must match feature_engineering.py) ---
# Assuming script is in PROJECT_ROOT/model/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data/mock_data')
VOCAB_PATH = os.path.join(DATA_DIR, 'feature_vocabularies.pkl')

# Data Split Paths
TRAIN_DATA_PATH = os.path.join(DATA_DIR, 'train_data_split.pkl')
VAL_DATA_PATH = os.path.join(DATA_DIR, 'val_data_split.pkl')
TEST_DATA_PATH = os.path.join(DATA_DIR, 'test_data_split.pkl')

# Model Saving Configuration
SAVED_MODELS_DIR = os.path.join(PROJECT_ROOT, 'saved_models') 
os.makedirs(SAVED_MODELS_DIR, exist_ok=True) # Ensure the directory exists

# --- Feature Order (CRITICAL for matching dataset structure to model inputs) ---
# This list must be in the exact order the Input layers are defined in the build_full_recommender_model
USER_FEATURE_NAMES = ['user_country', 'user_device_type', 'user_risk_score', 'days_since_first_trade', 'recent_views_count']
ITEM_FEATURE_NAMES = ['asset_ticker', 'asset_sector', 'market_cap_log', 'asset_volatility']
ALL_FEATURE_NAMES = USER_FEATURE_NAMES + ITEM_FEATURE_NAMES

# --- 1. Utility Functions ---

def load_data_split(path: str) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Loads feature and label data from a pickle file."""
    try:
        with open(path, 'rb') as f:
            data = pickle.load(f)
        return data['features'], data['labels']
    except FileNotFoundError:
        print(f"Error: Data file not found at {path}. Did you run the feature engineering script?")
        exit(1)


def load_vocabularies() -> Tuple[Dict, Dict]:
    """Loads feature specifications (vocabularies) from a pickle file."""
    try:
        with open(VOCAB_PATH, 'rb') as f:
            vocabs = pickle.load(f)
        return vocabs['user_specs'], vocabs['item_specs']
    except FileNotFoundError:
        print(f"Error: Vocabulary file not found at {VOCAB_PATH}. Did you run the feature engineering script?")
        exit(1)


def create_tf_dataset(features: Dict[str, np.ndarray], labels: np.ndarray, batch_size: int, shuffle: bool = True) -> tf.data.Dataset:
    """
    FIX: Creates a tf.data.Dataset using a tuple of features instead of a dictionary 
    to bypass C++ memory issues.
    """
    
    # 1. Extract the pre-processed NumPy arrays in the defined order
    feature_arrays = [features[name] for name in ALL_FEATURE_NAMES]
    
    # 2. Create a tuple of feature tensors (X) and the label tensor (Y)
    # The dataset will iterate over slices of this tuple: (feature_1, feature_2, ..., feature_N), label
    feature_tensors = tuple(tf.constant(arr, dtype=arr.dtype) for arr in feature_arrays)
    label_tensor = tf.constant(labels)

    # 3. Create the dataset from tensor slices
    dataset = tf.data.Dataset.from_tensor_slices((feature_tensors, label_tensor))
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(labels)) 
        
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

def preprocess_features(features: Dict[str, np.ndarray], specs: Dict) -> Dict[str, np.ndarray]:
    """
    Converts categorical string features to integer indices OUTSIDE the TensorFlow graph 
    for stable training. Also ensures numerical features are float32.
    """
    processed_features = features.copy()

    for feature_name, spec in specs.items():
        if spec['is_categorical']:
            vocab = spec['vocab']
            
            # Use temporary StringLookup layer for conversion
            lookup_layer = tf.keras.layers.StringLookup(
                vocabulary=vocab, 
                mask_token=None, 
                num_oov_indices=1,
                dtype=tf.string 
            )

            string_tensor = tf.constant(processed_features[feature_name], dtype=tf.string)
            indices_tensor = lookup_layer(string_tensor)

            # Replace the string array with int64 index array
            processed_features[feature_name] = indices_tensor.numpy().astype(np.int64)
        else:
            # Ensure all continuous features are float32
            processed_features[feature_name] = processed_features[feature_name].astype(np.float32)
            
    return processed_features


# --- 2. Model Definition: The Two Towers ---

def build_tower(specs: Dict, feature_names: List[str], tower_name: str) -> Tuple[Model, Dict[str, tf.Tensor]]:
    """
    Builds a single tower. 
    NOTE: Input order matters for the final Model construction to match the dataset tuple order.
    """
    
    # --- 2a. Define Input Layers in the correct, predefined order ---
    inputs = {}
    input_list = [] # List to hold the Keras Input layers in order

    for feature_name in feature_names:
        spec = specs[feature_name]
        
        if spec['is_categorical']:
            input_layer = Input(shape=(1,), name=feature_name, dtype=tf.int64)
        else:
            input_layer = Input(shape=(spec['shape'],), name=feature_name, dtype=tf.float32)
            
        inputs[feature_name] = input_layer
        input_list.append(input_layer)


    all_towers_outputs = [] # To hold the final embedding/output of each feature's path

    # --- 2b. Feature Processing (Iterating through the input list to maintain order) ---
    for x in input_list:
        feature_name = x.name.split('/')[0] # Get the name of the input layer
        spec = specs[feature_name]
        
        if spec['is_categorical']:
            vocab = spec['vocab']
            vocabulary_size = len(vocab) + 1 
            embedding_dim = max(4, int(np.sqrt(vocabulary_size)))

            embedding_layer = Embedding(
                input_dim=vocabulary_size, 
                output_dim=embedding_dim, 
                name=f'{tower_name}_{feature_name}_embedding'
            )

            x_squeezed = tf.squeeze(x, axis=1) 
            x_embed = embedding_layer(x_squeezed) 
            
            all_towers_outputs.append(x_embed)
            
        else:
            all_towers_outputs.append(x)

    # --- 2c. Concatenate and MLP ---
    if len(all_towers_outputs) > 1:
        x = Concatenate(axis=-1)(all_towers_outputs)
    else:
        x = all_towers_outputs[0]
    
    x = Dense(64, activation='relu', name=f'{tower_name}_dense_1')(x)
    output_vector = Dense(32, activation='relu', name=f'{tower_name}_embedding_output')(x)

    # Model is built using the ordered list of Input layers
    tower_model = Model(inputs=input_list, outputs=output_vector, name=f'{tower_name}_Tower')

    return tower_model, inputs

def build_full_recommender_model(user_specs: Dict, item_specs: Dict) -> Tuple[Model, Model, Model]:
    """Combines the User and Item towers for a full prediction model."""
    
    all_specs = {**user_specs, **item_specs}
    
    # 1. Build the separate towers
    # Pass the feature names to ensure Input layers are created in the right order
    user_tower, user_inputs_dict = build_tower(user_specs, USER_FEATURE_NAMES, 'user')
    item_tower, item_inputs_dict = build_tower(item_specs, ITEM_FEATURE_NAMES, 'asset')
    
    # 2. Combine all inputs in the ALL_FEATURE_NAMES order for the final model
    # Note: user_inputs_dict and item_inputs_dict are already ordered within themselves
    full_model_inputs_ordered = [user_inputs_dict[name] for name in USER_FEATURE_NAMES] + \
                                [item_inputs_dict[name] for name in ITEM_FEATURE_NAMES]
    
    # 3. Get the final embedding vectors
    user_embedding = user_tower(list(user_inputs_dict.values()))
    item_embedding = item_tower(list(item_inputs_dict.values()))
    
    # 4. Combine embeddings for a final prediction layer (Classification Head)
    combined_features = Concatenate(axis=-1)([user_embedding, item_embedding])
    
    # Classification Head (MLP for prediction)
    x = Dense(64, activation='relu', name='ranking_dense_1')(combined_features)
    x = Dense(32, activation='relu', name='ranking_dense_2')(x)
    prediction = Dense(1, activation='sigmoid', name='prediction_output')(x) 
    
    # Create the final model with the combined, ordered input list
    full_model = Model(inputs=full_model_inputs_ordered, outputs=prediction, name='TwoTowerRecommender')
    
    return full_model, user_tower, item_tower

# --- 3. Main Execution ---

if __name__ == '__main__':
    print("--- Starting Two-Tower Recommender Training ---")
    
    # --- A. Load Data and Specs ---
    user_specs, item_specs = load_vocabularies()
    
    # Load data splits
    train_features, train_labels = load_data_split(TRAIN_DATA_PATH)
    val_features, val_labels = load_data_split(VAL_DATA_PATH)

    # --- A.5 Pre-process Features (The Fix for the Malloc Error) ---
    print("Pre-processing categorical features (string -> index) and casting numerical features to float32...")
    all_specs = {**user_specs, **item_specs}
    train_features = preprocess_features(train_features, all_specs)
    val_features = preprocess_features(val_features, all_specs)
    print("Pre-processing complete. All features are now numerical (int64 or float32).")
    
    # --- B. Create tf.data.Dataset ---
    BATCH_SIZE = 1024 
    
    # Dataset creation now uses the tuple-based approach for stability
    train_ds = create_tf_dataset(train_features, train_labels, batch_size=BATCH_SIZE, shuffle=True)
    val_ds = create_tf_dataset(val_features, val_labels, batch_size=BATCH_SIZE, shuffle=False)
    
    # --- C. Build and Compile Model ---
    full_model, user_tower_model, asset_tower_model = build_full_recommender_model(user_specs, item_specs)
    
    print("\n--- Model Summary (Full Ranking Model) ---")
    full_model.summary()
    print("---------------------\n")

    # Set TensorFlow log level to suppress the verbose shuffle messages after this point
    # tf.get_logger().setLevel('ERROR') 

    model_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    full_model.compile(
        optimizer='adam',
        loss='binary_crossentropy', 
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')] 
    )
    
    # --- D. Train Model ---
    EPOCHS = 10 
    
    print(f"Starting training for {EPOCHS} epochs...")
    history = full_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        verbose=1,
        # Adding a callback just in case training stalls
        callbacks=[model_callback] 
    )
    
    print("\n--- Training Complete ---")
    
    # --- E. Save the trained model and towers for production deployment ---
    
    # 1. Save the full ranking model
    full_model_save_path = os.path.join(SAVED_MODELS_DIR, 'two_tower_ranking_model.h5')
    full_model.save(full_model_save_path) 
    print(f"Full Ranking Model saved to: {full_model_save_path}")
    
    # 2. Save the User Tower (for user embedding retrieval)
    user_tower_save_path = os.path.join(SAVED_MODELS_DIR, 'user_tower.h5')
    user_tower_model.save(user_tower_save_path)
    print(f"User Tower (Accepts INTs) saved to: {user_tower_save_path}")
    
    # 3. Save the Asset Tower (for asset embedding retrieval)
    asset_tower_save_path = os.path.join(SAVED_MODELS_DIR, 'asset_tower.h5')
    asset_tower_model.save(asset_tower_save_path)
    print(f"Asset Tower (Accepts INTs) saved to: {asset_tower_save_path}")
    
    # --- F. Print Note for Dataset Ordering ---
    print("\n--- Feature Order Check ---")
    print("The training dataset expects features in the exact tuple order:")
    for i, name in enumerate(ALL_FEATURE_NAMES):
        print(f"  {i+1}. {name}")
