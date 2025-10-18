import pandas as pd
import json
import os

# Define the directory where the mock feature files are stored
MOCK_FEATURE_DIR = "./data/mock_feature_store"

def load_training_data(feature_dir: str = MOCK_FEATURE_DIR) -> pd.DataFrame:
    """
    Loads user features, item features, and interaction logs,
    and joins them to create the final, model-ready training DataFrame.

    This function simulates the crucial ETL (Extract, Transform, Load) step
    of generating training data from an Offline Feature Store (like BigQuery/Snowflake).
    """
    print(f"Loading data from: {feature_dir}")

    # 1. Load the core datasets (simulating tables in a Feature Store)
    try:
        # Ground Truth: Interaction Log (The fact table we join to)
        interactions_df = pd.read_json(os.path.join(feature_dir, "interactions.json"))

        # Dimension Table: User Features
        users_df = pd.read_json(os.path.join(feature_dir, "users.json"))

        # Dimension Table: Item Features
        assets_df = pd.read_json(os.path.join(feature_dir, "assets.json"))

    except FileNotFoundError as e:
        print(f"Error: Required file not found. Ensure mock data is generated and saved to the correct directory: {e}")
        return pd.DataFrame()

    # 2. Join User Features to the Interaction Log
    # This is equivalent to a SQL JOIN on interactions_df.userId = users_df.userId
    training_data = pd.merge(
        interactions_df,
        users_df,
        on="userId",
        how="left" # Use left join to keep all interactions, even if a user is missing (shouldn't happen here)
    )

    # 3. Join Item Features (Assets) to the combined DataFrame
    # This is equivalent to a SQL JOIN on training_data.assetId = assets_df.assetId
    training_data = pd.merge(
        training_data,
        assets_df,
        on="assetId",
        how="left"
    )
    
    # 4. Final Data Preparation and Cleanup (Optional, but good practice)
    training_data = training_data.drop(columns=['interaction_type']) # Drop the raw type now that we have the binary 'target'
    
    print("\nTraining Data Generation Complete!")
    print(f"Total training examples generated: {len(training_data):,}")
    print(f"Columns (Features + Target): {list(training_data.columns)}")

    return training_data

if __name__ == '__main__':
    # Execute the loader and display a sample of the final dataset
    final_df = load_training_data()

    if not final_df.empty:
        print("\n--- SAMPLE OF FINAL MODEL-READY TRAINING DATA ---")
        print(final_df.head())
        print("\n--- DATA TYPES ---")
        print(final_df.dtypes)
