import logging
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Set
from services.url_candidate_detector import URLCandidate

# Configure high-fidelity local execution logging parameters
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("LinguaShield.URLReconstruction")

@dataclass(frozen=True)
class ReconstructedURLOption:
    """Dataclass holding a synthesized URL candidate alongside its structural plausibility score."""
    url: str
    plausibility_score: float
    hypothesis_type: str
    average_ocr_confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)

class URLReconstructionEngine:
    """Token-based structural assembler that probabilistically reconstructs 
    fractured text chunks into valid, scored URL path candidates.
    """
    
    def __init__(self):
        self.protocols: Set[str] = {"http", "https", "htttp", "httpss"}
        self.subdomains: Set[str] = {"www", "wvw", "vvv"}
        self.tlds: Set[str] = {"com", "net", "org", "gov", "edu", "xyz", "co", "io", "in", "biz", "info"}
        self.path_anchors: Set[str] = {"login", "verify", "verification", "secure", "account", "update", "pay"}

    def _classify_token_roles(self, candidate: URLCandidate) -> List[Tuple[str, Any]]:
        """Maps individual token sequence strings to functional URL component roles."""
        classified_tokens = []
        tld_found = False
        
        for token in candidate.tokens:
            text_lower = token.text.lower().strip()
            
            # Clean common terminal punctuation attached to individual tokens
            text_clean = re.sub(r'^[^\w]+|[^\w]+$', '', text_lower)
            
            if text_clean in self.protocols or "http" in text_clean:
                classified_tokens.append(("PROTOCOL", token))
            elif text_clean in self.subdomains:
                classified_tokens.append(("SUBDOMAIN", token))
            elif text_clean in self.tlds and not tld_found:
                classified_tokens.append(("TLD", token))
                tld_found = True  # The first matching structural TLD locks the domain edge
            elif tld_found or text_clean in self.path_anchors:
                classified_tokens.append(("PATH_PART", token))
            else:
                classified_tokens.append(("CORE_DOMAIN", token))
                
        return classified_tokens

    def reconstruct_candidate(self, candidate: URLCandidate) -> List[ReconstructedURLOption]:
        """Evaluates token roles, builds alternative structural layout hypotheses, 

        and assigns plausibility ranks to the generated URLs.
        """
        if not candidate.tokens:
            return []

        roles = self._classify_token_roles(candidate)
        avg_ocr_conf = sum(t.confidence for _, t in roles) / len(roles)
        
        # Isolate individual architectural elements based on role groups
        protocols_list = [t.text.lower().strip() for role, t in roles if role == "PROTOCOL"]
        subdomains_list = [t.text.lower().strip() for role, t in roles if role == "SUBDOMAIN"]
        core_domain_list = [t.text.lower().strip() for role, t in roles if role == "CORE_DOMAIN"]
        tlds_list = [t.text.lower().strip() for role, t in roles if role == "TLD"]
        paths_list = [t.text.lower().strip() for role, t in roles if role == "PATH_PART"]

        # Enforce structural defaults if essential structural components were dropped by the OCR
        protocol_str = "http://"
        if protocols_list:
            # Reconstruct common protocol drops securely
            base_proto = protocols_list[0]
            protocol_str = "https://" if "s" in base_proto else "http://"
            
        subdomain_str = "www." if subdomains_list else ""
        
        tld_str = ".com"
        if tlds_list:
            # Strip accidental punctuation read inside the TLD block
            tld_str = "." + re.sub(r'[^\w]', '', tlds_list[0])
            
        # Clean and filter core domain word fragments
        cleaned_core_parts = [re.sub(r'[^\w-]', '', p) for p in core_domain_list if p.strip()]
        
        # If the core domain list is empty, default to fallback to avoid empty hostnames
        if not cleaned_core_parts:
            cleaned_core_parts = ["secure-verification-panel"]

        # Extract and format trailing URL path elements
        cleaned_path_parts = [re.sub(r'[^\w-]', '', p) for p in paths_list if p.strip()]
        path_suffix = "/" + "/".join(cleaned_path_parts) if cleaned_path_parts else ""

        options: List[ReconstructedURLOption] = []

        # ---------------------------------------------------------------------
        # HYPOTHESIS ALPHA: Hyphenated Domain Chain (Standard Phishing Setup)
        # ---------------------------------------------------------------------
        hyp_alpha_hostname = "-".join(cleaned_core_parts)
        url_alpha = f"{protocol_str}{subdomain_str}{hyp_alpha_hostname}{tld_str}{path_suffix}"
        
        # High base plausibility score because scammers frequently combine words with hyphens
        score_alpha = avg_ocr_conf * 1.1 
        if subdomain_str: score_alpha += 5.0
        if tlds_list: score_alpha += 10.0
        
        options.append(ReconstructedURLOption(
            url=url_alpha,
            plausibility_score=min(score_alpha, 100.0),
            hypothesis_type="Hyphenated_Domain_Chain",
            average_ocr_confidence=avg_ocr_conf,
            metadata={"hostname": hyp_alpha_hostname, "tld": tld_str}
        ))

        # ---------------------------------------------------------------------
        # HYPOTHESIS BETA: Deep Subdomain Hierarchy
        # ---------------------------------------------------------------------
        hyp_beta_hostname = ".".join(cleaned_core_parts)
        url_beta = f"{protocol_str}{subdomain_str}{hyp_beta_hostname}{tld_str}{path_suffix}"
        
        # Slightly lower structural weight because deep dot-separated domains are less common
        score_beta = avg_ocr_conf * 0.95
        if subdomain_str: score_beta += 5.0
        if tlds_list: score_beta += 10.0
        
        options.append(ReconstructedURLOption(
            url=url_beta,
            plausibility_score=min(score_beta, 100.0),
            hypothesis_type="Deep_Subdomain_Hierarchy",
            average_ocr_confidence=avg_ocr_conf,
            metadata={"hostname": hyp_beta_hostname, "tld": tld_str}
        ))

        # Sort options so the highest-plausibility candidate sits at the top
        sorted_options = sorted(options, key=lambda o: o.plausibility_score, reverse=True)
        logger.info(f"[✓] Assembled URL candidates. Top Option: {sorted_options[0].url} [Score: {sorted_options[0].plausibility_score:.2f}]")
        return sorted_options