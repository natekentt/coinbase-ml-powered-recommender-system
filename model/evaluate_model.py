import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import pickle
import numpy as np
from typing import Dict, List, Tuple
from train_model import load_data_split, create_tf_dataset, preprocess_features, ALL_FEATURE_NAMES, USER_FEATURE_NAMES, ITEM_FEATURE_NAMES, load_vocabularies

# --- Configuration (Copied from train_model.py for consistency) ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data/mock_data')
TEST_DATA_PATH = os.path.join(DATA_DIR, 'test_data_split.pkl')
SAVED_MODELS_DIR = os.path.join(PROJECT_ROOT, 'saved_models') 

FULL_MODEL_PATH = os.path.join(SAVED_MODELS_DIR, 'two_tower_ranking_model.h5')


def evaluate_ranking_model():
    """
    Loads the full trained ranking model and evaluates its performance on the held-out test set.
    """
    print("--- Starting Model Evaluation on Test Data ---")

    # 1. Load Data and Specs
    print(f"Loading test data from: {TEST_DATA_PATH}")
    test_features, test_labels = load_data_split(TEST_DATA_PATH)
    user_specs, item_specs = load_vocabularies()

    # 2. Pre-process Features (Must match training script exactly)
    print("Pre-processing categorical features (string -> index) and casting numerical features...")
    all_specs = {**user_specs, **item_specs}
    test_features = preprocess_features(test_features, all_specs)
    print("Pre-processing complete.")

    # 3. Create tf.data.Dataset (Must match training script exactly)
    BATCH_SIZE = 1024 
    test_ds = create_tf_dataset(test_features, test_labels, batch_size=BATCH_SIZE, shuffle=False)

    # 4. Load the Full Model
    try:
        # Load the model using the custom objects required for complex layers
        model = load_model(
            FULL_MODEL_PATH, 
            custom_objects={
                'Concatenate': tf.keras.layers.Concatenate, 
                'Dense': tf.keras.layers.Dense,
                'Embedding': tf.keras.layers.Embedding
            },
            compile=True # Recompile the model with the same loss and metrics
        )
        print(f"\nSuccessfully loaded model from: {FULL_MODEL_PATH}")
    except Exception as e:
        print(f"\nERROR loading model: {e}")
        print("Ensure the model file exists and was saved successfully after training.")
        return

    # 5. Compile the loaded model with the required metrics (essential for evaluation)
    # The loaded model is already compiled, but we ensure the metrics are set.
    
    # 6. Evaluate the Model
    print("\n--- Evaluating Model ---")
    
    # Evaluate returns the loss first, then the metrics in the order defined in .compile()
    results = model.evaluate(
        test_ds, 
        verbose=1,
        return_dict=True
    )

    # 7. Print Results
    print("\n=============================================")
    print("        FINAL TEST SET EVALUATION")
    print("=============================================")
    print(f"Test Loss (Binary Cross-Entropy): {results.get('loss'):.4f}")
    print(f"Test Accuracy: {results.get('accuracy')*100:.2f}%")
    print(f"Test AUC (Area Under Curve): {results.get('auc'):.4f}")
    print("=============================================")

if __name__ == '__main__':
    # Ensure train_model.py utilities are accessible by making sure current directory is model/
    current_dir = os.path.basename(os.getcwd())
    if current_dir != 'model':
        print("Note: Running from outside the 'model' directory. Ensure paths are correct.")

    # Suppress verbose TensorFlow warnings during evaluation
    tf.get_logger().setLevel('ERROR')
    
    evaluate_ranking_model()
