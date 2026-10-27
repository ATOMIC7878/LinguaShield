import logging
import os
import re
import joblib
import pandas as pd
import numpy as np
from typing import List, Dict, Any

# Import the 8-module high-resilience URL processing framework layers
from services.image_enhancement import ImageEnhancementEngine
from services.ocr_engine import MultiEngineOCRCoordinator
from services.ocr_fusion import OCRFusionEngine
from services.url_candidate_detector import URLCandidateDetector
from services.url_reconstruction import URLReconstructionEngine
from services.ocr_error_repair import OCRErrorRepairEngine
from services.url_validation import URLValidationEngine
from services.feature_generator import URLFeatureGenerator

# Configure integrated high-fidelity logging parameters
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("LinguaShield.VisionOrchestrator")

# Establish relative file structural pathways for model loading
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "linguashield_classifier.pkl")

# Initialize all core engine services persistently in memory at application launch
enhancer = ImageEnhancementEngine()
ocr_coordinator = MultiEngineOCRCoordinator()
fusion_engine = OCRFusionEngine()
candidate_detector = URLCandidateDetector()
reconstruction_engine = URLReconstructionEngine()
repair_engine = OCRErrorRepairEngine()
validator = URLValidationEngine()
feature_generator = URLFeatureGenerator()

# Persistent Global Machine Learning Pipeline Holder
classifier_pipeline = None

if os.path.exists(MODEL_PATH):
    try:
        logger.info(f"[✓] Loading Pre-Trained Hybrid Scikit-Learn Model Core: {MODEL_PATH}")
        classifier_pipeline = joblib.load(MODEL_PATH)
    except Exception as e:
        logger.error(f"[!] Serialized model compilation error: {e}")
else:
    logger.warning(f"[!] Model snapshot not found at '{MODEL_PATH}'. Running in OCR text-extraction fallback mode.")

def extract_text_and_predict(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    """Master Pipeline Orchestrator that ingests image byte streams, executes 
    the 8-module recovery loop, structures feature spaces, and runs predictive 
    classifications to return a complete threat assessment JSON payload.
    """
    try:
        logger.info(f"--- Launching Integrated Threat Audit: {filename} ---")
        
        # 1. Image Enhancement Engine (Module 1 - Multi-Matrix Generation)
        variants = enhancer.process_source_bytes(file_bytes)
        
        # 2. Multi-Engine OCR coordinator (Module 2 - Parallel Local Passes)
        raw_tokens = ocr_coordinator.execute_all_engines(variants)
        
        # 3. Spatial Resolution Fusion (Module 3 - Rows clustered and welded)
        fused_tokens = fusion_engine.fuse_and_weld_tokens(raw_tokens)
        
        # 4. Token-Level Candidate Detector (Module 4 - Structural Word Scans)
        url_candidates = candidate_detector.detect_candidates(fused_tokens)
        
        # 5, 6, 7 & 8. Reconstruction, Error Repair, Validation, and Feature Generation Cycles
        processed_urls_pool: List[str] = []
        extracted_features_pool: List[Dict[str, Any]] = []
        
        for candidate in url_candidates:
            options = reconstruction_engine.reconstruct_candidate(candidate)
            if not options:
                continue
                
            top_option = options[0]
            repaired_output = repair_engine.execute_repair_cycle(top_option)
            
            validation_result = validator.validate_url_structure(
                repaired_output.repaired_url, 
                base_repair_confidence=repaired_output.repair_confidence
            )
            
            if validation_result.is_syntactically_valid:
                processed_urls_pool.append(validation_result.url)
                
                feature_vector = feature_generator.generate_features(
                    url_str=validation_result.url,
                    hostname=validation_result.hostname,
                    protocol=validation_result.protocol,
                    path_depth=validation_result.path_depth,
                    ocr_conf=top_option.average_ocr_confidence,
                    repair_conf=repaired_output.repair_confidence,
                    recon_conf=top_option.plausibility_score,
                    validation_score=validation_result.validation_score
                )
                extracted_features_pool.append(feature_vector.raw_feature_dict)

        # Compile the unified master spatial body text
        clean_master_text = fusion_engine.compile_to_master_string(fused_tokens)
        
        # ─── ADVERSARIAL SPACE-WELD FALLBACK GATEWAY ───
        # If token bounding boxes failed to cluster space-separated URLs, run text heuristic recovery
        if not processed_urls_pool:
            logger.info("[!] Staging heuristic fallback recovery sweep for space-separated links...")
            normalized_space_text = re.sub(r'\s+', ' ', clean_master_text.lower())
            
            # Scans text context for space-divided link sequences (e.g., 'http www split domain')
            fallback_regex = re.compile(r'(https?)\s+(www)\s+([a-z0-9\s\-]+)\s+(com|net|org|gov|xyz)\s+([a-z0-9\-]+)')
            match = fallback_regex.search(normalized_space_text)
            
            if match:
                proto, w3, core_labels, tld, path_route = match.groups()
                # Remove extra internal text spacings within host sub-labels
                clean_host_labels = ".".join([label.strip() for label in core_labels.split() if label.strip()])
                
                # Rebuild structural URL geometry layout
                reconstructed_fallback_url = f"{proto}://{w3}.{clean_host_labels}.{tld}/{path_route}"
                
                # Feed the recovered fallback URL through the validation layer
                fallback_validation = validator.validate_url_structure(reconstructed_fallback_url, base_repair_confidence=85.0)
                
                if fallback_validation.is_syntactically_valid:
                    processed_urls_pool.append(fallback_validation.url)
                    
                    fallback_features = feature_generator.generate_features(
                        url_str=fallback_validation.url,
                        hostname=fallback_validation.hostname,
                        protocol=fallback_validation.protocol,
                        path_depth=fallback_validation.path_depth,
                        ocr_conf=90.0,
                        repair_conf=85.0,
                        recon_conf=85.0,
                        validation_score=fallback_validation.validation_score
                    )
                    extracted_features_pool.append(fallback_features.raw_feature_dict)
                    logger.info(f"[✓ Heuristic Recovery Match] -> Welded link: {fallback_validation.url}")

        # Safely append recovered link patterns to tail to ensure regex compatibility
        if processed_urls_pool:
            unique_urls = list(set(processed_urls_pool))
            clean_master_text += " " + " ".join(unique_urls)

        # Default fallback feature vector if the screenshot lacks any web links
        selected_numeric_features = {
            "protocol_detected": 0, "number_of_dots": 0, "number_of_hyphens": 0,
            "hostname_length": 0, "path_depth": 0, "digit_count": 0, "entropy": 0.0,
            "ocr_confidence": 100.0, "repair_confidence": 100.0,
            "reconstruction_confidence": 100.0, "validation_score": 0.0,
            "contains_phishing_keywords": 1 if any(kw in clean_master_text.lower() for kw in ["free", "verify", "login", "pay"]) else 0
        }

        # If link features exist, pick the highest validation tracking profile for maximum risk checking
        if extracted_features_pool:
            selected_numeric_features = max(extracted_features_pool, key=lambda f: f.get("validation_score", 0.0))

        # Build complete record payload row combining text and isolated numeric properties
        inference_record = {"msg_content": clean_master_text}
        inference_record.update(selected_numeric_features)
        
        # Wrap row metrics inside a single-entry Pandas Dataframe matching model expectations
        inference_df = pd.DataFrame([inference_record])

        # Execute classification inference runs if the model file compiled properly
        verdict = "Model Unavailable"
        confidence_percentage = 0.0
        
        if classifier_pipeline is not None:
            raw_prediction = classifier_pipeline.predict(inference_df)[0]
            probabilities = classifier_pipeline.predict_proba(inference_df)[0]
            
            verdict = "Scam / Phishing" if raw_prediction == 1 else "Safe Chat"
            confidence_percentage = float(probabilities[raw_prediction] * 100.0)
            logger.info(f"[+] Prediction Computed Successfully. Verdict: {verdict} ({confidence_percentage:.2f}%)")
        else:
            logger.warning("Classification pass bypassed. Check that backend/models/linguashield_classifier.pkl exists.")

        # Structure complete JSON output schema ready for front-end consumption
        return {
            "status": "success",
            "filename": filename,
            "verdict": verdict,
            "confidence": round(confidence_percentage, 2),
            "extracted_text": clean_master_text,
            "recovered_urls": list(set(processed_urls_pool)),
            "structural_risk_metrics": {
                "hostname_length": int(selected_numeric_features["hostname_length"]),
                "number_of_dots": int(selected_numeric_features["number_of_dots"]),
                "number_of_hyphens": int(selected_numeric_features["number_of_hyphens"]),
                "character_entropy": float(selected_numeric_features["entropy"]),
                "validation_score": float(selected_numeric_features["validation_score"])
            }
        }

    except Exception as pipeline_fault:
        logger.error(f"[!] Critical internal failure inside Orchestrator Framework: {pipeline_fault}")
        return {
            "status": "error",
            "filename": filename,
            "verdict": "Processing Failure",
            "confidence": 0.0,
            "extracted_text": f"Error Logged: {str(pipeline_fault)}",
            "recovered_urls": [],
            "structural_risk_metrics": {}
        }