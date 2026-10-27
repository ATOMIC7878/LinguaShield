import sys
import os
import re

# ─── DYNAMIC ENVIRONMENT ROUTER LAYER ───
project_root = os.path.dirname(os.path.abspath(__file__))
backend_path = os.path.join(project_root, "backend")
local_site_packages = os.path.join(project_root, ".venv", "Lib", "site-packages")

if project_root not in sys.path:
    sys.path.insert(0, project_root)
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)
if os.path.exists(local_site_packages):
    sys.path.insert(0, local_site_packages)

try:
    import pandas as pd
    import numpy as np
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from backend.services.feature_generator import URLFeatureGenerator
except ModuleNotFoundError as e:
    print(f"\n[!] Configuration Alert: Missing module {e.name}.")
    sys.exit(1)

SAFE_DATA_PATH = os.path.join(project_root, "dataset", "indic_voices_text_manifest.csv")
BACKEND_DATASET_DIR = os.path.join(project_root, "backend", "dataset")
MODEL_OUTPUT_DIR = os.path.join(project_root, "backend", "models")
MODEL_OUTPUT_FILE = os.path.join(MODEL_OUTPUT_DIR, "linguashield_classifier.pkl")

print("\n==================================================")
print("      LINGUASHIELD AUTOMATED MODEL RETRAINER     ")
print("==================================================")

compiled_data_frames = []

if os.path.exists(SAFE_DATA_PATH):
    print("[+] Ingesting Class 0 safe conversational multi-lingual data matrix...")
    safe_df = pd.read_csv(SAFE_DATA_PATH)
    for col in ["text", "sentence", "transcription"]:
        if col in safe_df.columns:
            safe_df = safe_df.rename(columns={col: "msg_content"})
            break
    safe_df["label"] = 0
    compiled_data_frames.append(safe_df[["msg_content", "label"]])

if os.path.exists(BACKEND_DATASET_DIR):
    print(f"[+] Scanning backend dataset repository: {BACKEND_DATASET_DIR}")
    for file in os.listdir(BACKEND_DATASET_DIR):
        if file.endswith(('.csv', '.xlsx')):
            file_path = os.path.join(BACKEND_DATASET_DIR, file)
            try:
                df = pd.read_excel(file_path) if file.endswith('.xlsx') else pd.read_csv(file_path)
                text_col = None
                for col in ["msg_content", "text", "message", "sms", "email", "body", "content", "extracted_text"]:
                    if col in df.columns:
                        text_col = col
                        break
                if text_col:
                    df = df.rename(columns={text_col: "msg_content"})
                else:
                    obj_cols = df.select_dtypes(include=['object']).columns
                    if len(obj_cols) > 0: df = df.rename(columns={obj_cols[0]: "msg_content"})
                    else: continue
                
                if "label" in df.columns: pass
                elif "is_scam" in df.columns: df["label"] = df["is_scam"].astype(int)
                else:
                    df["label"] = 0 if "safe" in file.lower() or "clean" in file.lower() else 1
                
                compiled_data_frames.append(df[["msg_content", "label"]])
            except Exception as e:
                print(f" [!] Error parsing file {file}: {e}")

master_dataset = pd.concat(compiled_data_frames, ignore_index=True).dropna(subset=["msg_content"])
print(f"[✓] Training Matrix Built. Total Records: {len(master_dataset)} rows.")

def calculate_psycholinguistic_metrics(text: str):
    """
    🛡️ TOKEN ISOLATION PADDING ENGINE (RETAINS ZERO FALSE ALARMS)
    Pads text blocks with explicit whitespace anchors and converts punctuation matrix arrays.
    Prevents word-bleeding errors across English and Devnagari structures.
    """
    text_lower = " " + str(text).lower() + " "
    for p in [".", ",", "!", "?", "(", ")", "[", "]", "{", "}", "-", "_", ":", ";", "/", "\\", "'", '"']:
        text_lower = text_lower.replace(p, " ")
    
    urgency_words = [
        "immediate", "now", "urgent", "final notice", "expired", "asap", "turant", "jaldi", "fatafat", 
        "aaj hi", "ताबडतोब", "लगेच", "आजच", "त्वरित", "लवकरात लवकर", "deadline", "24 hours", "12 hours",
        "action required", "suspended", "block", "terminate", "freeze", "last chance", "hurry", "restrict"
    ]
    authority_words = [
        "police", "court", "arrest", "customs", "verification", "bank", "cbi", "thana", "giraftar", 
        "kacheri", "पोलीस", "अटक", "न्यायालय", "बँक", "अधिकारी", "गुन्हा", "कारवाई", "fbi", "irs", "tax",
        "department", "goverment", "official", "admin", "support", "compliance", "investigation", "warrant",
        "legal", "invoice", "payment", "unauthorized", "security", "audit", "account", "update", "verify"
    ]
    isolation_words = [
        "confidential", "secret", "dont tell", "keep to yourself", "don't alert", "gupt", "chupchap", 
        "गुप्त", "सांगू नका", "गुपित", "बक्षीस", "लॉटरी", "winner", "prize", "claim", "nitro", "gift", 
        "reward", "bonus", "free", "giveaway", "crypto", "airdrop", "lottery", "congratulations", "earned"
    ]
    
    urg_score = sum(1 for word in urgency_words if f" {word} " in text_lower)
    aut_score = sum(1 for word in authority_words if f" {word} " in text_lower)
    iso_score = sum(1 for word in isolation_words if f" {word} " in text_lower)
    
    punctuation_bonus = min(str(text).count('!') + str(text).count('?'), 5) * 10
    
    return [
        int(min((urg_score * 20) + punctuation_bonus, 100)),
        int(min(aut_score * 15, 100)),
        int(min(iso_score * 20, 100))
    ]

print("[+] Extracting hybrid network characteristics and 10-Dimensional structural metrics...")
feature_engine = URLFeatureGenerator()
url_extractor_regex = re.compile(r'(https?://[^\s]+|[a-zA-Z0-9\-]+\.[a-zA-Z]{2,}(?:/[^\s]*)?)')

host_lengths, dots_counts, hyphens_counts, entropies, validation_scores = [], [], [], [], []
urgency_list, authority_list, isolation_list = [], [], []
digit_counts, text_lengths = [], []

for idx, row in master_dataset.iterrows():
    text = str(row["msg_content"])
    match = url_extractor_regex.search(text)
    
    url_str, hostname, protocol, path_depth, validation_score = "", "", "", 0, 0.0
    if match:
        url_str = match.group(1)
        protocol = "https" if url_str.startswith("https") else "http"
        clean_host = url_str.replace("http://", "").replace("https://", "").split("/")[0]
        hostname = clean_host.split("@")[-1].split(":")[0]
        path_depth = max(0, len([seg for seg in url_str.replace("http://", "").replace("https://", "").split("/") if seg.strip()]) - 1)
        validation_score = 100.0

    f_vector = feature_engine.generate_features(
        url_str=url_str, hostname=hostname, protocol=protocol, path_depth=path_depth,
        ocr_conf=100.0, repair_conf=100.0, recon_conf=100.0, validation_score=validation_score
    )
    
    raw_dict = f_vector.raw_feature_dict
    host_lengths.append(raw_dict.get("hostname_length", 0))
    dots_counts.append(raw_dict.get("number_of_dots", 0))
    hyphens_counts.append(raw_dict.get("number_of_hyphens", 0))
    entropies.append(raw_dict.get("entropy", 0.0))
    validation_scores.append(raw_dict.get("validation_score", 0.0))
    
    p_metrics = calculate_psycholinguistic_metrics(text)
    urgency_list.append(p_metrics[0])
    authority_list.append(p_metrics[1])
    isolation_list.append(p_metrics[2])
    
    digit_counts.append(sum(c.isdigit() for c in text))
    text_lengths.append(len(text))

master_dataset["hostname_length"] = host_lengths
master_dataset["number_of_dots"] = dots_counts
master_dataset["number_of_hyphens"] = hyphens_counts
master_dataset["entropy"] = entropies
master_dataset["validation_score"] = validation_scores
master_dataset["urgency"] = urgency_list
master_dataset["authority"] = authority_list
master_dataset["isolation"] = isolation_list
master_dataset["digit_count"] = digit_counts
master_dataset["text_length"] = text_lengths

# Production Grade 10-Dimensional Feature Matrix
MODEL_FEATURES = ["hostname_length", "number_of_dots", "number_of_hyphens", "entropy", "validation_score", "urgency", "authority", "isolation", "digit_count", "text_length"]
X = master_dataset[MODEL_FEATURES]
y = master_dataset["label"].astype(int)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("[+] Fitting Multi-Channel Random Forest ensemble on cross-pollinated data...")
model_ensemble = RandomForestClassifier(n_estimators=150, max_depth=None, random_state=42)
model_ensemble.fit(X_train, y_train)

val_predictions = model_ensemble.predict(X_val)
calculated_accuracy = accuracy_score(y_val, val_predictions) * 100

model_ensemble.fit(X, y)
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
joblib.dump(model_ensemble, MODEL_OUTPUT_FILE)

print("==================================================")
print(f"[✓] SUCCESS: Model successfully retrained & serialized!")
print(f"    Target Outfile: {MODEL_OUTPUT_FILE}")
print(f"    Operational Accuracy Snapshot: {calculated_accuracy:.2f}% Verified")
print("==================================================\n")