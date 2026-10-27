import sys
import os

# ─── ENVIRONMENT ALIGNMENT ───
backend_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(backend_dir)

if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import asyncio
import joblib
import pandas as pd
from services.audio_speech import transcribe_audio_to_text
from services.feature_generator import URLFeatureGenerator
from services.feature_engineering import FinancialFraudEvaluator
from services.vocoder_analyzer import DeepfakeVocoderAnalyzer
from generate_test_audio import compile_threat_voice_samples

MODEL_PATH = os.path.join(backend_dir, "models", "linguashield_classifier.pkl")
MODEL_FEATURES = [
    "hostname_length", "number_of_dots", "number_of_hyphens", "entropy", 
    "validation_score", "urgency", "authority", "isolation", "digit_count", "text_length"
]

def calculate_psycholinguistic_vectors(text: str):
    text_lower = " " + str(text).lower() + " "
    for p in [".", ",", "!", "?", "(", ")", "[", "]", "{", "}", "-", "_", ":", ";", "/", "\\", "'", '"']:
        text_lower = text_lower.replace(p, " ")
    
    urgency_words = ["immediate", "now", "urgent", "expired", "asap", "deadline", "24 hours", "terminate"]
    authority_words = ["police", "court", "arrest", "cbi", "govt", "official", "compliance", "warning", "security"]
    isolation_words = ["confidential", "secret", "dont tell", "keep to yourself", "claim", "reward", "nitro"]
    
    urg_score = sum(1 for word in urgency_words if f" {word} " in text_lower)
    aut_score = sum(1 for word in authority_words if f" {word} " in text_lower)
    iso_score = sum(1 for word in isolation_words if f" {word} " in text_lower)
    
    return {
        "urgency": int(min((urg_score * 20), 100)),
        "authority": int(min(aut_score * 15, 100)),
        "isolation": int(min(iso_score * 20, 100))
    }

async def run_end_to_end_test():
    print("\n==========================================================")
    print("      LINGUASHIELD END-TO-END DEEPFAKE & THREAT TEST      ")
    print("==========================================================")
    
    # Step 1: Generate target synthetic audio samples
    print("\n[Step 1] Triggering Neural Voice Generator...")
    await compile_threat_voice_samples()
    
    # Step 2: Load Model
    print("\n[Step 2] Loading LinguaShield AI Model Core...")
    if not os.path.exists(MODEL_PATH):
        print(f"[!] Critical Error: Classifier model file missing at '{MODEL_PATH}'.")
        return
        
    clf_model = joblib.load(MODEL_PATH)
    print("[✓] Model bound successfully.")
    
    # Step 3: Initialize Engine Services
    feature_engine = URLFeatureGenerator()
    fraud_evaluator = FinancialFraudEvaluator()
    vocoder_analyzer = DeepfakeVocoderAnalyzer()
    
    samples_dir = os.path.join(backend_dir, "test_audio_samples")
    test_files = ["ai_cloned_bank_scam.mp3", "ai_gaming_nitro_lure.mp3"]
    
    # Step 4: Run Audits against test files
    for filename in test_files:
        filepath = os.path.join(samples_dir, filename)
        if not os.path.exists(filepath):
            print(f"[!] Warning: Test sample {filename} not found.")
            continue
            
        print("\n" + "=" * 60)
        print(f" 🎧 RUNNING MULTI-MODAL AUDIT: {filename}")
        print("=" * 60)
        
        with open(filepath, "rb") as f:
            file_bytes = f.read()
            
        # 1. Acoustic Vocoder Analysis
        print("[+] Executing Deepfake Acoustic Vocoder Analysis...")
        acoustic_res = vocoder_analyzer.analyze_audio_bytes(file_bytes, filename)
        
        # 2. Speech-to-Text Transcription
        print("[+] Transcribing Speech Audio Stream...")
        extracted_text = transcribe_audio_to_text(file_bytes, filename)
        print(f"    ↳ Extracted Text: \"{extracted_text}\"")
        
        # 3. Feature Extraction
        vectors = calculate_psycholinguistic_vectors(extracted_text)
        fraud_metrics = fraud_evaluator.extract_lure_metrics(extracted_text)
        
        feature_record = {
            "hostname_length": 0, "number_of_dots": 0, "number_of_hyphens": 0,
            "entropy": 0.0, "validation_score": 0.0,
            "urgency": vectors["urgency"],
            "authority": vectors["authority"],
            "isolation": vectors["isolation"],
            "digit_count": sum(c.isdigit() for c in extracted_text),
            "text_length": len(extracted_text)
        }
        
        inference_df = pd.DataFrame([feature_record])[MODEL_FEATURES]
        
        # 4. Model Prediction
        prediction = int(clf_model.predict(inference_df)[0])
        confidence_score = float(clf_model.predict_proba(inference_df)[0][prediction]) * 100
        
        # 5. Proactive Override logic
        if acoustic_res["is_synthetic_voice"]:
            prediction = 1
            confidence_score = max(confidence_score, float(acoustic_res["vocoder_score"]))
            
        if fraud_metrics["compound_financial_lure_score"] >= 65.0:
            prediction = 1
            confidence_score = max(confidence_score, float(fraud_metrics["compound_financial_lure_score"]))
            
        is_scam = (prediction == 1)
        
        # 6. Display Telemetry Report
        print("\n📊 --- ACOUSTIC & THREAT FORENSIC REPORT ---")
        print(f" • Primary Status      : {'🚨 CRITICAL THREAT' if is_scam else '🟢 LOW RISK'}")
        print(f" • Confidence Score     : {confidence_score:.2f}%")
        print(f" • Classification Tag   : {'DEEPFAKE VOICE CLONE DETECTED' if acoustic_res['is_synthetic_voice'] else 'NATURAL / TEXT THREAT'}")
        print(f" • Vocoder Synthetic %  : {acoustic_res['vocoder_score']}%")
        print(f" • Spectral High-Freq % : {acoustic_res.get('high_freq_ratio', 0)}")
        print(f" • Frame Energy Jitter  : {acoustic_res.get('energy_jitter', 0)}")
        print(f" • Financial Lure Score : {fraud_metrics['compound_financial_lure_score']}%")
        print(f" • Urgency Vector       : {vectors['urgency']}%")
        print(f" • Authority Vector     : {vectors['authority']}%")

if __name__ == "__main__":
    asyncio.run(run_end_to_end_test())