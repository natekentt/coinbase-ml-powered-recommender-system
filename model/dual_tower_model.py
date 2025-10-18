import tensorflow as tf
import os # Added os import for environment variable fix
from tensorflow.keras.layers import Layer, Input, Dense, Embedding, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.layers.experimental.preprocessing import StringLookup, TextVectorization, CategoryEncoding
from typing import Dict, Tuple, List

# --- FIX for 'Illegal instruction: 4' ---
# This disables the oneDNN CPU optimizations which often trigger low-level 
# instruction errors on older or specific processors.
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' 
# ----------------------------------------

# --- Configuration (Must match feature_engineering.py) ---
EMBEDDING_DIM = 32 # The final dimension for the output of both towers (the vector size)
MAX_TOKENS = 10000 
SEQUENCE_LENGTH = 32

# Define feature groups for clarity
USER_CATEGORICAL_FEATURES = ['region', 'risk_score', 'account_tier', 'favorite_sector']
ITEM_CATEGORICAL_FEATURES = ['ticker', 'sector']
ITEM_NUMERICAL_FEATURES = ['market_cap_usd', 'volatility_index']
ITEM_TEXT_FEATURE = 'asset_description_text'

# --- 1. Keras Preprocessing Layer Definitions ---
# These functions define the Keras preprocessing layers needed to transform 
# raw string inputs into dense numerical inputs or embeddings.

def create_string_lookup_layer(feature_name: str, vocabulary: List[str]) -> Layer:
    """
    Creates a StringLookup layer to map categorical strings to unique integer IDs.
    The last two elements of the vocabulary are reserved for unknown/masked tokens.
    """
    return StringLookup(
        vocabulary=vocabulary,
        mask_token=None,
        num_oov_indices=1, # Number of out-of-vocabulary indices
        name=f"lookup_{feature_name}"
    )

def create_text_vectorization_layer(vocabulary: List[str]) -> Layer:
    """
    Creates a TextVectorization layer to tokenize and map the asset description
    text to integer sequences, constrained by MAX_TOKENS and SEQUENCE_LENGTH.
    """
    return TextVectorization(
        max_tokens=MAX_TOKENS,
        output_sequence_length=SEQUENCE_LENGTH,
        vocabulary=vocabulary,
        name="vectorize_asset_text"
    )

# --- 2. Model Tower Construction ---

def build_tower(
    name: str, 
    input_specs: Dict[str, Tuple[int, List[str]]],
    output_dimension: int
) -> Model:
    """
    Constructs a single tower (User or Item) using Keras functional API.

    Args:
        name: Name of the tower (e.g., 'user' or 'item').
        input_specs: Dictionary where keys are feature names and values are 
                     (input_shape, optional_vocabulary_list).
        output_dimension: The size of the final vector embedding (EMBEDDING_DIM).

    Returns:
        A compiled Keras Model instance for the tower.
    """
    input_tensors = {}
    encoded_features = []

    for feature_name, (shape, vocabulary) in input_specs.items():
        # 1. Define the Input Tensor
        dtype = tf.string if vocabulary else tf.float32
        input_tensors[feature_name] = Input(shape=(shape,), name=f"{name}_{feature_name}", dtype=dtype)
        current_layer = input_tensors[feature_name]
        
        # 2. Handle Categorical Features (StringLookup -> Embedding)
        if feature_name in USER_CATEGORICAL_FEATURES + ITEM_CATEGORICAL_FEATURES:
            # Map strings to integer IDs
            lookup_layer = create_string_lookup_layer(feature_name, vocabulary)
            integer_ids = lookup_layer(current_layer)
            
            # Embed the IDs into a dense vector space (e.g., size 4-10)
            # Embedding size is typically 4th root of vocab size.
            vocab_size = len(vocabulary) + 1 
            embedding_size = max(4, int(vocab_size**0.25))
            
            feature_embedding = Embedding(
                input_dim=vocab_size, 
                output_dim=embedding_size, 
                name=f"{name}_embed_{feature_name}"
            )(integer_ids)
            
            # Flatten the embedding output since the input shape is (batch, 1)
            encoded_features.append(tf.keras.layers.Flatten()(feature_embedding))

        # 3. Handle Numerical Features (Pass-Through)
        elif feature_name in ITEM_NUMERICAL_FEATURES:
            # Numerical features (already normalized) are just passed through
            # and concatenated directly. They need to be cast to float32 if they were string inputs
            encoded_features.append(tf.cast(current_layer, tf.float32))

        # 4. Handle Text Feature (TextVectorization -> Embedding -> GlobalAveragePooling)
        elif feature_name == ITEM_TEXT_FEATURE:
            # Text must be vectorized (tokenized and converted to IDs)
            vectorize_layer = create_text_vectorization_layer(vocabulary)
            integer_sequences = vectorize_layer(current_layer)
            
            # Embed the integer sequences
            text_embedding = Embedding(
                input_dim=MAX_TOKENS,
                output_dim=EMBEDDING_DIM, # Use the final embedding dim for text
                name=f"{name}_embed_text"
            )(integer_sequences)
            
            # Summarize the sequence embedding into a single vector
            pooled_embedding = tf.keras.layers.GlobalAveragePooling1D()(text_embedding)
            encoded_features.append(pooled_embedding)

    # 3. Concatenate all encoded features
    if len(encoded_features) > 1:
        combined_features = Concatenate(name=f"{name}_concat")(encoded_features)
    else:
        combined_features = encoded_features[0]

    # 4. Final MLP Layers (The "Tower" part)
    # The MLP processes the concatenated features down to the final embedding dimension.
    x = Dense(64, activation='relu', name=f"{name}_dense_1")(combined_features)
    x = Dense(output_dimension, activation=None, name=f"{name}_output_vector")(x)
    
    # 5. Model Output and Normalization
    # L2 normalization ensures all vectors lie on a unit sphere, which is crucial 
    # for distance metrics like Cosine Similarity.
    output_vector = tf.nn.l2_normalize(x, axis=1, name=f"{name}_embedding")
    
    return Model(inputs=input_tensors, outputs=output_vector, name=f"{name}_tower")


def create_dual_tower_model(
    user_input_specs: Dict[str, Tuple[int, List[str]]],
    item_input_specs: Dict[str, Tuple[int, List[str]]],
    embedding_dimension: int = EMBEDDING_DIM
) -> Model:
    """
    Creates and compiles the complete Dual-Tower Model (two separate towers).
    
    The final model is defined by multiplying the two tower outputs to get a
    single prediction score (the similarity).
    """
    
    # Build the individual towers
    user_tower = build_tower("user", user_input_specs, embedding_dimension)
    item_tower = build_tower("item", item_input_specs, embedding_dimension)
    
    # The final combined model takes all user and item inputs
    all_inputs = {**user_tower.input, **item_tower.input}
    
    # Get the final normalized vectors
    user_vector = user_tower(user_tower.input)
    item_vector = item_tower(item_tower.input)
    
    # The final prediction is the dot product of the two vectors (similarity score)
    # This is the standard way to combine Dual-Tower outputs for ranking/retrieval models.
    dot_product = tf.reduce_sum(user_vector * item_vector, axis=1, keepdims=True)
    
    # Create the final trainable model
    combined_model = Model(
        inputs=all_inputs, 
        outputs=dot_product, 
        name="Dual_Tower_Retriever"
    )
    
    # Compilation for training with a binary classification objective
    combined_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), # Use from_logits=True as the output is not sigmoid-activated
        metrics=[tf.keras.metrics.AUC(name='auc')]
    )
    
    print("\n--- Dual-Tower Model Architecture Created ---")
    user_tower.summary()
    item_tower.summary()
    
    return combined_model

if __name__ == '__main__':
    # This block requires feature_engineering.py to run first to get the vocabularies.
    # We will fully integrate this in the next step, but here are the mock specs:
    
    print("This file defines the architecture only. The actual training step will be in 'model/train_model.py'.")

    # Mock Vocabularies (in reality, feature_engineering.py would generate these)
    MOCK_VOCAB_USER = {
        'region': (1, ['US', 'EU', 'APAC']),
        'risk_score': (1, ['LOW', 'MEDIUM', 'HIGH']),
        'account_tier': (1, ['Retail', 'Pro']),
        'favorite_sector': (1, ['DeFi', 'Gaming', 'Layer 1', 'AI']),
    }
    
    MOCK_VOCAB_ITEM = {
        'ticker': (1, [f'COIN_{i}' for i in range(1, 100)]),
        'sector': (1, ['DeFi', 'Gaming', 'Layer 1', 'AI']),
        'market_cap_usd': (1, []), # Numerical, no vocab
        'volatility_index': (1, []), # Numerical, no vocab
        'asset_description_text': (1, ["mock token", "decentralized", "project", "gaming", "defi"]),
    }
    
    # combined_model = create_dual_tower_model(MOCK_VOCAB_USER, MOCK_VOCAB_ITEM)
    # print("\nCombined Model Summary:")
    # combined_model.summary()
