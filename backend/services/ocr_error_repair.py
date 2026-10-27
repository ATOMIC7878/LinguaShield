import logging
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple
from services.url_reconstruction import ReconstructedURLOption

# Configure high-fidelity local execution logging parameters
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("LinguaShield.OCRErrorRepair")

@dataclass(frozen=True)
class RepairedURLOutput:
    """Dataclass holding the final error-corrected URL string along with structural metric traces."""
    original_url: str
    repaired_url: str
    repair_confidence: float
    mutations_applied: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

class OCRErrorRepairEngine:
    """Context-aware character confusion engine that dynamically resolves pixel-level 
    OCR syntax anomalies within URL components.
    """
    
    def __init__(self):
        # Establish structural alphanumeric target replacement matrices
        self.tld_corrections: Dict[str, str] = {
            "c0m": "com", "c0rn": "com", "n3t": "net", "0rg": "org", 
            "g0v": "gov", "x1z": "xyz", "p1c": "pic", "1n": "in"
        }
        
        # Core visual confusion ruleset maps
        self.visual_confusion_map: List[Tuple[str, str, str]] = [
            ("vv", "w", "Double-V to W Fusion"),
            ("rn", "m", "R-N to M Bleed Fix"),
            ("cl", "d", "C-L to D Character Welds"),
            ("1uck", "luck", "Digit-1 to L Substitution"),
            ("w1n", "win", "Digit-1 to I Substitution")
        ]

    def _repair_protocol(self, protocol_chunk: str, mutations: List[str]) -> str:
        """Enforces clean, strict protocol structures, repairing common numeric substitutions."""
        chunk = protocol_chunk.lower().strip()
        
        if "http5" in chunk or "https" in chunk:
            if "http5" in chunk:
                mutations.append("Protocol: Fixed '5' to 's' anomaly")
            return "https://"
            
        if "ht1p" in chunk or "http" in chunk:
            if "ht1p" in chunk:
                mutations.append("Protocol: Fixed '1' to 't' anomaly")
            return "http://"
            
        return protocol_chunk

    def _repair_tld(self, tld_chunk: str, mutations: List[str]) -> str:
        """Validates and clean-corrects Top-Level Domain segment modifications."""
        clean_tld = tld_chunk.lower().replace(".", "").strip()
        
        # Check standard dictionary matrix overrides
        if clean_tld in self.tld_corrections:
            corrected = self.tld_corrections[clean_tld]
            mutations.append(f"TLD: Corrected dynamic extension variant '.{clean_tld}' to '.{corrected}'")
            return f".{corrected}"
            
        # Address common digit-to-character flips within custom domains
        fixed_tld = clean_tld.replace("0", "o").replace("1", "i").replace("3", "e")
        if fixed_tld != clean_tld:
            mutations.append(f"TLD: Flattened character matrix filters from '.{clean_tld}' to '.{fixed_tld}'")
            return f".{fixed_tld}"
            
        return tld_chunk

    def _repair_hostname_core(self, hostname: str, mutations: List[str]) -> str:
        """Applies advanced morphological text matrix fixes to clear layout confusions in domains."""
        repaired = hostname
        
        # Loop over character shape mappings sequentially
        for target_typo, clean_char, label in self.visual_confusion_map:
            if target_typo in repaired:
                repaired = repaired.replace(target_typo, clean_char)
                mutations.append(f"Hostname: Applied matrix transform [{label}]")
                
        # Fix mixed character-digit boundaries (e.g., 'bank0f' -> 'bankof')
        # Standard rules target explicit contextual anomalies like '0' inside purely alphabetical blocks
        if re.search(r'[a-zA-Z]0[a-zA-Z]', repaired):
            repaired = re.sub(r'([a-zA-Z])0([a-zA-Z])', r'\1o\2', repaired)
            mutations.append("Hostname: Contextual replacement of '0' to 'o'")
            
        return repaired

    def execute_repair_cycle(self, reconstructed_option: ReconstructedURLOption) -> RepairedURLOutput:
        """Parses the reconstructed URL option components, executes contextual error 

        corrections, and computes updated structural confidence rankings.
        """
        original = reconstructed_option.url
        mutations_applied: List[str] = []
        
        # Breakdown URL layout zones via structured regex splits
        url_pattern = re.compile(r'^(https?://)?([^/]+)(.*)$', re.IGNORECASE)
        match = url_pattern.match(original)
        
        if not match:
            return RepairedURLOutput(
                original_url=original, repaired_url=original,
                repair_confidence=reconstructed_option.plausibility_score,
                mutations_applied=[]
            )
            
        protocol, full_host, path_suite = match.groups()
        protocol = protocol if protocol else "http://"
        path_suite = path_suite if path_suite else ""
        
        # Further isolate the TLD extension boundary marker from the hostname host
        tld_match = re.search(r'(\.[a-zA-Z0-9\-\_]+)$', full_host)
        if tld_match:
            tld_str = tld_match.group(1)
            core_host = full_host[:-len(tld_str)]
        else:
            tld_str = ".com"
            core_host = full_host

        # ---------------------------------------------------------------------
        # CONTEXTUAL COMPONENT PROCESSING LOOP
        # ---------------------------------------------------------------------
        repaired_proto = self._repair_protocol(protocol, mutations_applied)
        repaired_tld = self._repair_tld(tld_str, mutations_applied)
        repaired_host = self._repair_hostname_core(core_host, mutations_applied)
        
        # Recombine component blocks into a clean, complete URL structure
        final_repaired_url = f"{repaired_proto}{repaired_host}{repaired_tld}{path_suite}"
        
        # Calculate dynamic repair confidence modifiers based on changes made
        base_confidence = reconstructed_option.plausibility_score
        penalty_score = len(mutations_applied) * 3.5 # Moderate boost for resolving structural anomalies
        final_confidence = min(base_confidence + penalty_score, 100.0)

        logger.info(f"[✓] Error Repair complete. Changes Tracker: {len(mutations_applied)} mutations applied.")
        return RepairedURLOutput(
            original_url=original,
            repaired_url=final_repaired_url,
            repair_confidence=final_confidence,
            mutations_applied=mutations_applied,
            metadata={
                "base_hypothesis": reconstructed_option.hypothesis_type,
                "had_structural_mutations": len(mutations_applied) > 0
            }
        )