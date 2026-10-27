import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

def train_lingua_shield_model(dataset_path: str, model_output_path: str):
    print("--- Initializing LinguaShield Classifier Training Run ---")
    
    # 1. Load your prototype historical threat registers
    df = pd.read_csv(dataset_path)
    
    # 🌟 THE CRITICAL VECTOR UPDATE
    # Expanded features to include the new greed-based vector alongside the fear-based ones
    feature_columns = [
        'temporal_urgency_cascade',
        'institutional_coercion',
        'social_isolation_injection',
        'compound_financial_lure_score'  # <-- New feature integrated here
    ]
    
    target_column = 'is_scam'  # Binary target classification (1 = Threat, 0 = Safe)
    
    # Ensure all required features exist in the dataset
    missing_cols = [col for col in feature_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing required feature columns: {missing_cols}")
        
    # 2. Extract Feature Matrix (X) and Target Vector (y)
    X = df[feature_columns].values
    y = df[target_column].values
    
    print(f"[Dataset Matrix Loaded] Total Samples: {X.shape[0]} | Total Features: {X.shape[1]}")
    
    # 3. Stratified Train-Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 4. Instantiate and Configure the Random Forest Classifier
    # We increase n_estimators slightly to give the trees room to map out the new feature paths
    clf = RandomForestClassifier(
        n_estimators=150,
        max_depth=8,
        min_samples_split=4,
        random_state=42,
        class_weight='balanced'  # Critical for addressing highly unbalanced scam datasets
    )
    
    # 5. Execute Model Fitting
    print("[Model Training] Fitting Random Forest trees against expanded feature space...")
    clf.fit(X_train, y_train)
    
    # 6. Evaluate Model Metrics against Test Splits
    y_pred = clf.predict(X_test)
    
    print("\n=== Model Evaluation Performance ===")
    print(classification_report(y_test, y_pred, target_names=["Cleared", "Critical Threat"]))
    
    # 🌟 FEATURE IMPORTANCE PROFILE
    print("=== Feature Importance Distribution ===")
    importances = clf.feature_importances_
    for name, importance in zip(feature_columns, importances):
        print(f"  - {name}: {importance * 100:.2f}%")
        
    # 7. Serialize and Save both the Model and Feature Architecture
    model_payload = {
        "classifier": clf,
        "feature_names": feature_columns
    }
    
    joblib.dump(model_payload, model_output_path)
    print(f"\n[✓ Training Complete] Updated model binary saved securely to: {model_output_path}")

if __name__ == "__main__":
    import os
    
    # 🌟 DYNAMIC PATH RESOLUTION: Get the absolute directory of train_classifier.py (which is backend/)
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Map paths dynamically relative to where this script lives
    DATA_DIR = os.path.join(SCRIPT_DIR, "data")
    MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
    
    # Auto-create the data and models folders if they don't exist yet
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    target_csv_path = os.path.join(DATA_DIR, "threat_registers.csv")
    target_model_path = os.path.join(MODELS_DIR, "random_forest_v2.pkl")
    
    # 🌟 EMERGENCY SAFETY NET: If the CSV file is missing, create a dummy placeholder 
    # so your prototype loop doesn't crash and you can see it work!
    if not os.path.exists(target_csv_path):
        print(f"[!] Warning: {target_csv_path} not found. Creating a baseline dummy dataset for verification...")
        dummy_data = {
            'temporal_urgency_cascade': [20, 0, 80, 10, 0, 90],
            'institutional_coercion': [15, 0, 70, 5, 0, 85],
            'social_isolation_injection': [0, 0, 50, 0, 0, 60],
            'compound_financial_lure_score': [0, 95, 10, 0, 100, 5], # Includes your new feature
            'is_scam': [1, 1, 1, 0, 1, 0] # 1 = Scam, 0 = Safe
        }
        # Duplicate the rows slightly so the Random Forest model has enough split data
        df_dummy = pd.DataFrame(dummy_data)
        df_large = pd.concat([df_dummy] * 10, ignore_index=True) 
        df_large.to_csv(target_csv_path, index=False)
        print(f"[✓] Baseline dummy data generated successfully at: {target_csv_path}")

    # Run the training execution loop with robust absolute paths
    train_lingua_shield_model(
        dataset_path=target_csv_path,
        model_output_path=target_model_path
    )