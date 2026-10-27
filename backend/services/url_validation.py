import logging
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse

# Configure high-fidelity local execution logging parameters
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("LinguaShield.URLValidation")

@dataclass(frozen=True)
class ValidatedURLResult:
    """Dataclass holding structural validation metrics and the finalized target URL string."""
    url: str
    is_syntactically_valid: bool
    validation_score: float  # Scale from 0.0 (Impossible syntax) to 100.0 (Flawless RFC layout)
    protocol: str
    hostname: str
    tld: str
    path_depth: int
    validation_anomalies: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

class URLValidationEngine:
    """Offline cryptographic and syntax validator that verifies the structural 
    integrity of recovered phishing URLs using strict RFC compliance checks.
    """
    
    def __init__(self):
        # Known common top-level domain extension structures for quick sanity filtering
        self.common_tlds = {"com", "net", "org", "gov", "edu", "xyz", "co", "io", "in", "biz", "info", "me", "cc", "tv"}
        # Regular expression validating legal label character maps (Alphanumeric + Hyphen)
        self.valid_label_pattern = re.compile(r'^[a-zA-Z0-9\-]+$')

    def validate_url_structure(self, url_string: str, base_repair_confidence: float = 100.0) -> ValidatedURLResult:
        """Parses the recovered URL without network dependencies, flags structural 
        anomalies, and returns an absolute syntax integrity score.
        """
        anomalies: List[str] = []
        score = 100.0
        
        # ─── THE FIXED EARLY-EXIT GUARD BLOCK ───
        if not url_string or not url_string.strip():
            return ValidatedURLResult(
                url="", is_syntactically_valid=False, validation_score=0.0,
                protocol="", hostname="", tld="", path_depth=0,
                validation_anomalies=["Input URL string is completely empty"]
            )

        try:
            # 1. Execute initial standard decomposition using urllib mapping libraries
            parsed = urlparse(url_string.strip())
            
            protocol = parsed.scheme.lower()
            hostname = parsed.netloc.lower()
            path = parsed.path
            
            # --- PROTOCOL VALIDATION ---
            if protocol not in ["http", "https"]:
                anomalies.append(f"Invalid Scheme Protocol: '{protocol}' is outside secure web scopes")
                score -= 30.0
                
            # --- HOSTNAME VALIDATION ---
            if not hostname:
                anomalies.append("Missing Hostname: URL structure lacks a valid network location identity block")
                return ValidatedURLResult(
                    url=url_string, is_syntactically_valid=False, validation_score=0.0,
                    protocol=protocol, hostname="", tld="", path_depth=0, validation_anomalies=anomalies
                )
                
            # Catch raw spacing leaks or illegal symbol inclusions inside the network block
            if any(char in hostname for char in [" ", "\\", "@", ":"]):
                anomalies.append("Malformed Netloc Block: Discovered spacing, slashes, or credential leaks inside host")
                score -= 25.0

            # --- RFC 1035 DOMAIN LABEL VALIDATION ---
            # Total domain name length must not exceed 253 characters
            if len(hostname) > 253:
                anomalies.append("RFC Violation: Hostname sequence length exceeds absolute 253 character cap")
                score -= 20.0

            labels = hostname.split(".")
            
            # Catch consecutive dots (e.g., 'company..com') which yields empty label blocks
            if "" in labels:
                anomalies.append("Syntax Fragmentation: Detected empty domain labels or consecutive dot sequences")
                score -= 40.0
                # Filter out empty entries to continue scanning remaining components
                labels = [label for label in labels if label]

            tld = ""
            if labels:
                tld = labels[-1]
                # A numeric TLD is an absolute configuration failure for standard web resolution
                if tld.isdigit():
                    anomalies.append(f"Fatal TLD Anomaly: Top-level extension segment '{tld}' cannot be purely numeric")
                    score -= 35.0
                elif tld not in self.common_tlds:
                    # Minor warning score penalty for exotic or unverified custom extensions
                    score -= 5.0

            # Scan individual subdomain and domain labels
            for label in labels:
                if len(label) > 63:
                    anomalies.append(f"RFC Label Violation: Component chunk '{label[:10]}...' length exceeds 63 character limit")
                    score -= 15.0
                if label.startswith("-") or label.endswith("-"):
                    anomalies.append(f"Label Geometry Fault: Component chunk '{label}' starts or ends with an illegal hyphen")
                    score -= 20.0
                if not self.valid_label_pattern.match(label):
                    anomalies.append(f"Illegal Characters Detected: Component chunk '{label}' contains illegal non-alphanumeric punctuation")
                    score -= 25.0

            # --- PATH ANALYSIS ---
            # Standardize path depth evaluations by counting split route marks
            path_segments = [seg for seg in path.split("/") if seg.strip()]
            depth = len(path_segments)

            # Integrate original OCR error-repair history metrics into the score weightings
            final_calculated_score = max(0.0, min(score, 100.0))
            weighted_integrity_score = (final_calculated_score * 0.6) + (base_repair_confidence * 0.4)
            
            is_valid = weighted_integrity_score >= 50.0 and "Missing Hostname" not in anomalies

            logger.info(f"[✓] Syntax Validation Complete. URL: {url_string} -> Score: {weighted_integrity_score:.2f}")
            return ValidatedURLResult(
                url=url_string,
                is_syntactically_valid=is_valid,
                validation_score=weighted_integrity_score,
                protocol=protocol,
                hostname=hostname,
                tld=tld,
                path_depth=depth,
                validation_anomalies=anomalies,
                metadata={"raw_syntax_score": final_calculated_score, "anomaly_count": len(anomalies)}
            )

        except Exception as validation_fault:
            logger.error(f"[!] Processing failure inside Validation Pipeline: {validation_fault}")
            return ValidatedURLResult(
                url=url_string, is_syntactically_valid=False, validation_score=0.0,
                protocol="", hostname="", tld="", path_depth=0,
                validation_anomalies=[f"Internal parser crash error: {str(validation_fault)}"]
            )