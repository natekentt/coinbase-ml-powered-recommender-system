import json
import random
import uuid
from datetime import datetime, timedelta
import os # Imported for file path handling

# --- Configuration for Mock Data Generation ---
# Set the number of entities for generation
N_USERS = 100
N_ASSETS = 50
N_INTERACTIONS = 5000  # Total number of implicit interactions to generate
OUTPUT_DIR = "./data/mock_feature_store" # Define the local directory to store the mock feature data

# Feature definitions aligning with the schemas
ASSET_SECTORS = ["DeFi", "Layer 1", "Gaming", "Stablecoin", "AI"]
USER_REGIONS = ["US", "EU", "APAC"]
RISK_SCORES = ["LOW", "MEDIUM", "HIGH"]
ACCOUNT_TIERS = ["Retail", "Pro"]
INTERACTION_TYPES = ["VIEW", "CLICK", "TRADE_SMALL", "TRADE_LARGE"]

START_DATE = datetime(2024, 1, 1)

def generate_asset_data(n_assets):
    """Generates synthetic Item Feature Sets (Coinbase Entities)."""
    assets = []
    asset_ids = [str(uuid.uuid4())[:8] for _ in range(n_assets)]

    for i, asset_id in enumerate(asset_ids):
        sector = random.choice(ASSET_SECTORS)
        
        # Simple text description for the NLP embedding layer (aligned with item_features.json)
        description = f"Highly competitive decentralized project focusing on {sector} infrastructure and governance."
        if sector == "Stablecoin":
            description = "A robust digital asset pegged to the US dollar for reliable store-of-value functionality."

        assets.append({
            "assetId": asset_id,
            "ticker": f"COIN_{i}",
            "sector": sector,
            "market_cap_usd": round(random.uniform(10_000_000, 100_000_000_000), 2),
            "volatility_index": round(random.uniform(0.1, 0.9), 3),
            "asset_description_text": description
        })
    return assets, asset_ids

def generate_user_data(n_users):
    """Generates synthetic User Feature Sets (User Profiles)."""
    users = []
    user_ids = [str(uuid.uuid4())[:12] for _ in range(n_users)]
    
    # Map to simulate the user's intrinsic preference/bias
    user_bias = {uid: random.choice(ASSET_SECTORS) for uid in user_ids}

    for user_id in user_ids:
        users.append({
            "userId": user_id,
            "region": random.choice(USER_REGIONS),
            "risk_score": random.choice(RISK_SCORES),
            "account_tier": random.choice(ACCOUNT_TIERS),
            "favorite_sector": user_bias[user_id] # Feature for modeling user bias
        })
    return users, user_ids, user_bias

def generate_interaction_data(user_ids, assets, user_bias, n_interactions):
    """Generates the Interaction Log (Training Ground Truth)."""
    interactions = []
    
    asset_map = {asset['assetId']: asset for asset in assets}

    for _ in range(n_interactions):
        user_id = random.choice(user_ids)
        user_favorite_sector = user_bias[user_id]

        # Bias the selection: 70% chance to pick an asset from the user's favorite sector
        if random.random() < 0.7:
            biased_assets = [a for a in assets if a['sector'] == user_favorite_sector]
            asset = random.choice(biased_assets) if biased_assets else random.choice(assets)
        else:
            # 30% chance for random interaction
            asset = random.choice(assets)

        # Generate a random timestamp within the defined period
        random_days = random.randint(0, 30)
        timestamp = START_DATE + timedelta(days=random_days, hours=random.randint(0, 23), minutes=random.randint(0, 59))

        # Weight interaction types: VIEW is most common, TRADE_LARGE is least
        interaction_type = random.choices(
            INTERACTION_TYPES,
            weights=[40, 30, 20, 10], 
            k=1
        )[0]

        # Define positive target (1) for strong implicit feedback, 0 otherwise (aligned with interaction_log.json)
        is_positive_interaction = interaction_type in ["CLICK", "TRADE_SMALL", "TRADE_LARGE"]

        interactions.append({
            "userId": user_id,
            "assetId": asset['assetId'],
            "timestamp": timestamp.isoformat(),
            "interaction_type": interaction_type,
            "target": 1 if is_positive_interaction else 0
        })
        
    return interactions

def generate_mock_data():
    """Main function to orchestrate data generation."""
    assets, asset_ids = generate_asset_data(N_ASSETS)
    users, user_ids, user_bias = generate_user_data(N_USERS)
    interactions = generate_interaction_data(user_ids, assets, user_bias, N_INTERACTIONS)

    return {
        "users": users,
        "assets": assets,
        "interactions": interactions
    }

def save_mock_data(mock_data, output_dir):
    """Saves the generated mock data into separate JSON files."""
    # Ensure the directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    file_paths = {}
    
    # Save Users (User Feature Set)
    user_path = os.path.join(output_dir, "users.json")
    with open(user_path, 'w') as f:
        json.dump(mock_data["users"], f, indent=2)
    file_paths["users"] = user_path

    # Save Assets (Item Feature Set)
    asset_path = os.path.join(output_dir, "assets.json")
    with open(asset_path, 'w') as f:
        json.dump(mock_data["assets"], f, indent=2)
    file_paths["assets"] = asset_path

    # Save Interactions (Ground Truth Log)
    interactions_path = os.path.join(output_dir, "interactions.json")
    with open(interactions_path, 'w') as f:
        json.dump(mock_data["interactions"], f, indent=2)
    file_paths["interactions"] = interactions_path
    
    return file_paths


if __name__ == '__main__':
    mock_data = generate_mock_data()
    
    # Save data to files
    saved_files = save_mock_data(mock_data, OUTPUT_DIR)
    
    print("--- MOCK FEATURE STORE DATA SAMPLE ---")
    # Print a sample of the structured mock data output in JSON format
    print(json.dumps({
        "users": mock_data["users"][:3],
        "assets": mock_data["assets"][:3],
        "interactions": mock_data["interactions"][:5]
    }, indent=2))

    print(f"\nSuccessfully generated {len(mock_data['users'])} users, {len(mock_data['assets'])} assets, and {len(mock_data['interactions'])} interactions.")
    print(f"Data saved to the '{OUTPUT_DIR}' directory as: ")
    for key, path in saved_files.items():
        print(f"  - {path}")
