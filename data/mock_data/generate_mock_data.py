import pandas as pd
import numpy as np
import os
import random

# --- Configuration ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'mock_data')
os.makedirs(DATA_DIR, exist_ok=True)

NUM_USERS = 5000
NUM_ASSETS = 100
NUM_INTERACTIONS = 500000  # Total rows for the final dataset

# --- 1. User Data Generation ---

def generate_user_data(n_users: int) -> pd.DataFrame:
    """Generates mock user features."""
    countries = ['US', 'CA', 'GB', 'DE', 'AU', 'IN', 'JP', 'BR', 'MX', 'SG']
    devices = ['mobile', 'desktop', 'tablet']
    
    data = {
        'user_id': [f'user_{i}' for i in range(n_users)],
        'user_country': np.random.choice(countries, n_users),
        'user_device_type': np.random.choice(devices, n_users, p=[0.6, 0.3, 0.1]),
        # Continuous numerical features
        'user_risk_score': np.random.uniform(1, 10, n_users).round(2), # 1 (low risk) to 10 (high risk)
        'days_since_first_trade': np.random.poisson(365, n_users),
        'recent_views_count': np.random.randint(0, 50, n_users)
    }
    return pd.DataFrame(data)

# --- 2. Item Data Generation ---

def generate_asset_data(n_assets: int) -> pd.DataFrame:
    """Generates mock asset (item) features."""
    sectors = ['DeFi', 'Layer1', 'Gaming', 'AI', 'Stablecoin', 'Web3', 'Meme']
    
    # Generate unique tickers
    tickers = [f'ASSET{i}' for i in range(1, n_assets + 1)]
    
    data = {
        'asset_id': [f'asset_{i}' for i in range(n_assets)],
        'asset_ticker': tickers,
        'asset_sector': np.random.choice(sectors, n_assets, p=[0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.2]),
        # Continuous numerical features
        'market_cap_log': np.random.uniform(10, 15, n_assets).round(2), # Log market cap
        'asset_volatility': np.random.uniform(0.01, 0.15, n_assets).round(4)
    }
    return pd.DataFrame(data)

# --- 3. Interaction Data & Merging ---

def generate_interaction_data(users_df: pd.DataFrame, assets_df: pd.DataFrame, n_interactions: int) -> pd.DataFrame:
    """
    Creates the final merged dataset with a binary target interaction column.
    """
    # Sample user_id and asset_id for interactions
    interaction_data = pd.DataFrame({
        'user_id': np.random.choice(users_df['user_id'], n_interactions),
        'asset_id': np.random.choice(assets_df['asset_id'], n_interactions)
    })

    # Generate a binary target interaction label (e.g., 1=purchase/watchlist, 0=view)
    # Simulate a bias: higher user risk scores and lower volatility assets might have a slightly higher chance of interaction
    # This keeps the model training task realistic
    
    # Merge with user features
    df = pd.merge(interaction_data, users_df, on='user_id', how='left')
    
    # Merge with asset features
    df = pd.merge(df, assets_df, on='asset_id', how='left')
    
    # Generate a pseudo-realistic target label
    # Probability of interaction based on features
    # NOTE: These weights are completely arbitrary for mocking data
    
    # Assign a baseline probability (e.g., 5% overall interaction rate)
    baseline_prob = 0.05
    
    # Adjust probability based on mock features
    # High risk score user is more likely to interact (+0.01 per point of risk score)
    df['prob_boost'] = (df['user_risk_score'] - 1) / 9 * 0.1 
    
    # Low volatility asset is slightly less likely to be ignored
    df['prob_boost'] -= (df['asset_volatility'] / 0.15) * 0.05
    
    # Cap probability
    df['interaction_prob'] = baseline_prob + df['prob_boost']
    df['interaction_prob'] = df['interaction_prob'].clip(lower=0.01, upper=0.20)
    
    # Generate the binary label
    df['target_interaction'] = (np.random.rand(n_interactions) < df['interaction_prob']).astype(int)
    
    # Drop intermediate columns
    df = df.drop(columns=['interaction_prob', 'prob_boost'])
    
    return df

# --- Main Execution ---

if __name__ == '__main__':
    print("--- Starting Mock Data Generation ---")
    
    # 1. Generate User and Asset tables
    users_df = generate_user_data(NUM_USERS)
    assets_df = generate_asset_data(NUM_ASSETS)
    
    print(f"Generated {len(users_df)} users and {len(assets_df)} assets.")

    # 2. Generate Interactions and Merge
    final_df = generate_interaction_data(users_df, assets_df, NUM_INTERACTIONS)
    
    # 3. Save the final merged data as CSV
    OUTPUT_FILE_PATH = os.path.join(DATA_DIR, 'mock_coinbase_data.csv')
    final_df.to_csv(OUTPUT_FILE_PATH, index=False)
    
    print(f"\nSuccessfully created and saved merged dataset to: {OUTPUT_FILE_PATH}")
    print(f"Total interaction samples: {len(final_df)}")
    print(f"Target interaction rate: {final_df['target_interaction'].mean():.4f}")
    
    print("\n--- Mock Data Generation Complete. Ready for Feature Engineering! ---")
