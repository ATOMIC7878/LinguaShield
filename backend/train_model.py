import logging
import os
import random
import re
import pandas as pd
import numpy as np
import joblib
from typing import List, Dict, Any, Tuple
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

# Import the Module 8 Feature Generator directly from your existing services architecture
from services.feature_generator import URLFeatureGenerator

# Configure production-grade execution logging parameters
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("LinguaShield.ModelTrainer")

class AdversarialDataPreparer:
    """Handles dataset loading, formatting alignment, and unmasking token placeholders 
    to mirror raw, high-fidelity OCR engine runtime outputs.
    """
    
    def __init__(self):
        # Simulated phishing links matching high-risk patterns
        self.malicious_urls = [
            "http://free-discord-nitro.xyz/claim",
            "https://steam-community-gifts.net/login",
            "http://royale-pass-giveaway.co/verify",
            "https://verification-panel-staff.info",
            "http://refund-tax-gov-portal.xyz/login"
        ]
        
        # Safe standard links
        self.safe_urls = [
            "https://github.com/trending",
            "https://wikipedia.org/wiki/Main_Page",
            "https://google.com",
            "https://docs.python.org/3/"
        ]
        
        self.malicious_invites = ["discord.gg/nitro-drop", "discord.gg/claim-rewards"]
        self.safe_invites = ["discord.gg/python", "discord.gg/gaming-lounge"]

    def unmask_placeholders(self, text: str, label: int) -> str:
        """Replaces static placeholder tokens (<URL>, <DISCORD_INVITE>) with dynamic, 
        realistic URL strings to prevent the model from hyper-focusing on plain text tokens.
        """
        if not isinstance(text, str):
            return ""
            
        processed_text = text
        
        # 1. Unmask URL references based on threat context
        if "<URL>" in processed_text:
            chosen_url = random.choice(self.malicious_urls) if label == 1 else random.choice(self.safe_urls)
            processed_text = processed_text.replace("<URL>", chosen_url)
            
        # 2. Unmask Discord Server Invites
        if "<DISCORD_INVITE>" in processed_text:
            chosen_invite = random.choice(self.malicious_invites) if label == 1 else random.choice(self.safe_invites)
            processed_text = processed_text.replace("<DISCORD_INVITE>", chosen_invite)
            
        # 3. Clean remaining low-signal markers
        processed_text = processed_text.replace("<EMOJI>", "😊").replace("<USER>", "@gamer_profile")
        
        return processed_text

class ContextualFeatureExtractor:
    """Computes and injects numeric URL structural profiles directly into the 
    training matrix for every row in the text dataset.
    """
    
    def __init__(self):
        self.feature_engine = URLFeatureGenerator()
        
        # Relaxed regex to locate embedded or recovered links within text frames
        self.url_extractor_regex = re.compile(r'(https?://[^\s]+|[a-zA-Z0-9\-]+\.[a-zA-Z]{2,}(?:/[^\s]*)?)')

    def extract_row_features(self, text: str) -> Dict[str, float]:
        """Scans a text line, isolates any present link segments, and computes 
        Module 8 structural metrics. Returns a baseline vector if no links exist.
        """
        match = self.url_extractor_regex.search(text)
        
        url_str = ""
        hostname = ""
        protocol = ""
        path_depth = 0
        
        if match:
            url_str = match.group(1)
            # Basic fallback segmentation for standalone training vectors
            protocol = "https" if url_str.startswith("https") else "http"
            clean_host = url_str.replace("http://", "").replace("https://", "").split("/")[0]
            hostname = clean_host.split("@")[-1].split(":")[0]
            path_depth = len([seg for seg in url_str.replace("http://", "").replace("https://", "").split("/") if seg.strip()]) - 1
            path_depth = max(0, path_depth)

        # Generate standard Module 8 features. 
        # OCR trust parameters are filled at 100% capacity during initial baseline training.
        vector = self.feature_engine.generate_features(
            url_str=url_str,
            hostname=hostname,
            protocol=protocol,
            path_depth=path_depth,
            ocr_conf=100.0,
            repair_conf=100.0,
            recon_conf=100.0,
            validation_score=100.0 if hostname else 0.0
        )
        
        return vector.raw_feature_dict

def train_and_serialize_model(dataset_dir: str, output_model_path: str):
    """Executes the complete machine learning engineering lifecycle: data ingestion from 
    multiple source tracks, feature compilation, training, evaluation, and serialization.
    """
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Target dataset directory not found at path: {dataset_dir}")

    logger.info(f"--- Launching LinguaShield Multi-Dataset Ingestion Process from: {dataset_dir} ---")
    
    csv_files = [f for f in os.listdir(dataset_dir) if f.endswith('.csv') and f != 'staging_feedback.csv']
    if not csv_files:
        raise FileNotFoundError(f"No valid CSV training dataset assets found in directory: {dataset_dir}")
        
    df_list = []
    
    for file_name in csv_files:
        file_path = os.path.join(dataset_dir, file_name)
        logger.info(f"[+] Processing dataset track: {file_name}")
        
        # Gracefully parse messy bytes from different platform sources
        temp_df = pd.read_csv(file_path, encoding='utf-8', encoding_errors='replace')
        
        # Clean white spaces from header row arrays if present
        temp_df.columns = [str(c).strip() for c in temp_df.columns]
        
        # ─── THE POLYMORPHIC SCHEMA ADAPTER LAYER ───
        text_column_resolved = None
        label_column_resolved = None
        
        # Dynamic text feature key mappings
        for candidate in ['msg_content', 'text_combined', 'text', 'v2']:
            if candidate in temp_df.columns:
                text_column_resolved = candidate
                break
                
        # Dynamic tracking label key mappings
        for candidate in ['label', 'v1']:
            if candidate in temp_df.columns:
                label_column_resolved = candidate
                break
                
        if not text_column_resolved or not label_column_resolved:
            logger.warning(f"[!] Skipping {file_name}: Structural verification keys could not be resolved.")
            continue
            
        # Isolate target metrics and normalize naming alignment
        sub_df = temp_df[[label_column_resolved, text_column_resolved]].copy()
        sub_df = sub_df.rename(columns={text_column_resolved: 'msg_content', label_column_resolved: 'label'})
        
        # Drop entries missing text signals completely
        sub_df = sub_df.dropna(subset=['msg_content'])
        
        # Normalize and map multi-class string labels (like UCI spam/ham) to clear binary vectors
        sub_df['label'] = sub_df['label'].astype(str).str.strip().str.lower()
        sub_df['label'] = sub_df['label'].replace({
            'ham': '0', 'safe': '0', '0': '0', '0.0': '0',
            'spam': '1', 'scam': '1', 'phishing': '1', '1': '1', '1.0': '1'
        })
        
        # Drop rows with non-binary variations remaining in the collection
        sub_df = sub_df[sub_df['label'].isin(['0', '1'])]
        sub_df['label'] = sub_df['label'].astype(int)
        
        logger.info(f"    ↳ Successfully ingested {sub_df.shape[0]} valid data matrix rows.")
        df_list.append(sub_df)
        
    if not df_list:
        raise ValueError("Ingestion failure: None of the target CSV assets contained valid operational layouts.")
        
    # Weld all datasets together into a single unified matrix frame
    df = pd.concat(df_list, ignore_index=True)
    logger.info(f"[✓] Multi-dataset fusion complete. Total combined rows: {df.shape[0]}")

    # Step 1: Adversarial Unmasking Preprocessing Pass
    preparer = AdversarialDataPreparer()
    df['msg_content'] = df.apply(lambda row: preparer.unmask_placeholders(row['msg_content'], row['label']), axis=1)
    logger.info("[✓] Token unmasking step complete. Placeholder vectors expanded.")

    # Step 2: Extract Hybrid Structural Matrix Features
    extractor = ContextualFeatureExtractor()
    logger.info("Compiling numerical URL structural matrices from chat rows...")
    
    feature_records = []
    for text in df['msg_content']:
        feature_records.append(extractor.extract_row_features(text))
        
    features_df = pd.DataFrame(feature_records)
    
    # Merge text strings and numeric metrics into a single training dataframe block
    training_master_df = pd.concat([df['msg_content'], features_df], axis=1)
    target_labels = df['label'].astype(int)

    # Step 3: Split Data for Local Validation Testing
    X_train, X_test, y_train, y_test = train_test_split(
        training_master_df, target_labels, test_size=0.20, random_state=42, stratify=target_labels
    )
    logger.info(f"Data partitioned successfully. Training size: {X_train.shape[0]} rows. Test validation size: {X_test.shape[0]} rows.")

    # Step 4: Build the Scikit-Learn Multimodal Column Transformer Pipeline
    numeric_feature_names = list(feature_records[0].keys())
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('text_tfidf', TfidfVectorizer(max_features=12000, ngram_range=(1, 2)), 'msg_content'),
            ('numeric_meta', 'passthrough', numeric_feature_names)
        ]
    )

    # Wrap the components inside a unified, secure execution Pipeline with balanced weights
    clf_pipeline = Pipeline(steps=[
        ('feature_processing_gate', preprocessor),
        ('classifier_core', RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1, class_weight="balanced"))
    ])

    # Step 5: Train the Unified Classifier Core
    logger.info("Executing local model fitting sweeps...")
    clf_pipeline.fit(X_train, y_train)
    logger.info("[✓] Model optimization complete.")

    # Step 6: Evaluate Model Accuracy and Metrics
    predictions = clf_pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    logger.info(f"--- Training Suite Metrics Report ---")
    logger.info(f"Absolute Validation Accuracy Score: {accuracy * 100:.2f}%")
    print(classification_report(y_test, predictions, target_names=["Ham / Safe Chat", "Scam / Phishing"]))

    # Step 7: Serialize and Export the Model Artifact
    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
    joblib.dump(clf_pipeline, output_model_path)
    logger.info(f"[✓] Serialized production pipeline model file successfully exported to: {output_model_path}")

if __name__ == "__main__":
    OUTPUT_PICKLE_PATH = os.path.join(os.path.dirname(__file__), "models", "linguashield_classifier.pkl")
    DATASET_DIR = os.path.join(os.path.dirname(__file__), "dataset")
    
    if not os.path.exists(DATASET_DIR):
        if os.path.exists(os.path.join("backend", "dataset")):
            DATASET_DIR = os.path.join("backend", "dataset")
        elif os.path.exists("dataset"):
            DATASET_DIR = "dataset"
        else:
            DATASET_DIR = r"C:\Users\ROHIT\Desktop\Codes\LinguaShield\backend\dataset"
            
    train_and_serialize_model(DATASET_DIR, OUTPUT_PICKLE_PATH)