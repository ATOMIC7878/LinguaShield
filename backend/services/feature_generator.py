import logging
import math
import re
from dataclasses import dataclass, field
from typing import Dict, Any, List

# Configure integrated local logging parameters
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("LinguaShield.FeatureGenerator")

@dataclass(frozen=True)
class PhishingFeatureVector:
    """Dataclass encapsulating comprehensive numeric and structural features 
    for direct ingestion into Scikit-Learn machine learning pipelines.
    """
    protocol_detected: int  # 1 if http/https found, 0 otherwise
    number_of_dots: int
    number_of_hyphens: int
    hostname_length: int
    path_depth: int
    digit_count: int
    entropy: float
    ocr_confidence: float
    repair_confidence: float
    reconstruction_confidence: float
    validation_score: float
    contains_phishing_keywords: int
    raw_feature_dict: Dict[str, Any] = field(default_factory=dict, repr=False)

class URLFeatureGenerator:
    """Advanced feature engineering engine that extracts rich cryptographic, 
    structural, and confidence vectors from recovered URL validation structures.
    """
    
    def __init__(self):
        self.phishing_keywords = {"login", "verify", "secure", "account", "refund", "bank", "update", "payment"}

    def _calculate_shannon_entropy(self, text: str) -> float:
        """Computes the Shannon Entropy of a text string to measure character 
        randomness, exposing algorithmically generated phishing domains.
        """
        if not text:
            return 0.0
            
        text_len = len(text)
        frequencies = {}
        for char in text:
            frequencies[char] = frequencies.get(char, 0) + 1
            
        entropy = 0.0
        for count in frequencies.values():
            probability = count / text_len
            entropy -= probability * math.log2(probability)
            
        return round(entropy, 4)

    def generate_features(
        self, 
        url_str: str, 
        hostname: str, 
        protocol: str,
        path_depth: int,
        ocr_conf: float, 
        repair_conf: float, 
        recon_conf: float, 
        validation_score: float
    ) -> PhishingFeatureVector:
        """Processes recovered URL layers, extracts comprehensive metric data vectors, 
        and maps them into a clean feature space payload.
        """
        logger.info(f"[✓] Compiling rich numeric feature vector space profile for host: '{hostname}'")
        
        # 1. Structural Metric Counting
        num_dots = hostname.count(".")
        num_hyphens = hostname.count("-")
        digits = sum(1 for char in hostname if char.isdigit())
        
        # 2. String Semantics
        has_protocol = 1 if protocol in ["http", "https"] else 0
        
        url_lower = url_str.lower()
        has_keywords = 1 if any(kw in url_lower for kw in self.phishing_keywords) else 0
        
        # 3. Cryptographic Character Evaluation
        string_entropy = self._calculate_shannon_entropy(hostname)
        
        # Build raw dict structure for quick unpacking into dataframes
        feature_map = {
            "protocol_detected": has_protocol,
            "number_of_dots": num_dots,
            "number_of_hyphens": num_hyphens,
            "hostname_length": len(hostname),
            "path_depth": path_depth,
            "digit_count": digits,
            "entropy": string_entropy,
            "ocr_confidence": ocr_conf,
            "repair_confidence": repair_conf,
            "reconstruction_confidence": recon_conf,
            "validation_score": validation_score,
            "contains_phishing_keywords": has_keywords
        }
        
        return PhishingFeatureVector(
            protocol_detected=has_protocol,
            number_of_dots=num_dots,
            number_of_hyphens=num_hyphens,
            hostname_length=len(hostname),
            path_depth=path_depth,
            digit_count=digits,
            entropy=string_entropy,
            ocr_confidence=ocr_conf,
            repair_confidence=repair_conf,
            reconstruction_confidence=recon_conf,
            validation_score=validation_score,
            contains_phishing_keywords=has_keywords,
            raw_feature_dict=feature_map
        )