import logging
import re
from dataclasses import dataclass, field
from typing import List, Set, Dict, Any
from services.ocr_fusion import FusedTextToken, OCRBoundingBox

# Configure high-fidelity local execution logging parameters
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("LinguaShield.URLCandidateDetector")

@dataclass
class URLCandidate:
    """Dataclass holding a clustered sequence of suspected URL tokens 

    alongside descriptive weight anchors for reconstruction mapping.
    """
    tokens: List[FusedTextToken]
    raw_joined_text: str
    base_anchor_score: float
    contains_protocol: bool
    contains_tld: bool
    metadata: Dict[str, Any] = field(default_factory=dict)

class URLCandidateDetector:
    """Token-level heuristic scanner that detects and isolates potential 

    phishing link clusters without relying on structural punctuation markers.
    """
    
    def __init__(self):
        # Establish structural alphanumeric target validation vocabularies
        self.protocols: Set[str] = {"http", "https", "htttp", "httpss"}
        self.subdomains: Set[str] = {"www", "wvw", "vvv"}
        self.tlds: Set[str] = {"com", "net", "org", "gov", "edu", "xyz", "co", "io", "in", "biz", "info"}
        
        self.phishing_keywords: Set[str] = {
            "login", "verify", "verification", "payment", "secure", "account", 
            "refund", "bank", "update", "profile", "billing", "dues", "online"
        }

    def _score_token_semantics(self, text: str) -> float:
        """Evaluates an individual string token and outputs a baseline score 

        based on its matching semantic anchors.
        """
        lower_token = text.lower().strip()
        score = 0.0
        
        # 1. Evaluate explicit routing anchors
        if lower_token in self.protocols or any(p in lower_token for p in ["http://", "https://"]):
            score += 50.0
        if lower_token in self.subdomains or lower_token.startswith("www."):
            score += 40.0
            
        # 2. Evaluate top-level domain extension structures
        if lower_token in self.tlds or any(f".{t}" in lower_token for t in self.tlds):
            score += 30.0
            
        # 3. Evaluate context markers
        if lower_token in self.phishing_keywords:
            score += 15.0
            
        # 4. Extract structural link shapes (handles partially read strings like 'powerbill-pay')
        if "/" in lower_token or "." in lower_token or "-" in lower_token:
            score += 10.0
            
        return score

    def group_tokens_by_line(self, tokens: List[FusedTextToken]) -> List[List[FusedTextToken]]:
        """Organizes fused tokens into grouped row indexes based on shared bounding metrics."""
        if not tokens:
            return []
            
        # Sort tokens primarily by top-left Y box coordinates to track structural rows
        sorted_tokens = sorted(tokens, key=lambda t: (t.bbox.top_left[1], t.bbox.top_left[0]))
        
        lines: List[List[FusedTextToken]] = []
        current_line: List[FusedTextToken] = []
        
        # Row tracking relies on a 20px spatial bounding threshold check
        for token in sorted_tokens:
            if not current_line:
                current_line.append(token)
                continue
                
            last_token = current_line[-1]
            if abs(token.bbox.top_left[1] - last_token.bbox.top_left[1]) <= 20:
                current_line.append(token)
            else:
                lines.append(sorted(current_line, key=lambda t: t.bbox.top_left[0]))
                current_line = [token]
                
        if current_line:
            lines.append(sorted(current_line, key=lambda t: t.bbox.top_left[0]))
            
        return lines

    def detect_candidates(self, fused_tokens: List[FusedTextToken]) -> List[URLCandidate]:
        """Scans the running token array, clusters groups that match structural 

        anchors, and isolates candidate sequences for reconstruction.
        """
        if not fused_tokens:
            return []
            
        token_lines = self.group_tokens_by_line(fused_tokens)
        candidates: List[URLCandidate] = []
        
        for line_index, line in enumerate(token_lines):
            i = 0
            n = len(line)
            
            while i < n:
                token = line[i]
                token_score = self._score_token_semantics(token.text)
                
                # If a token exhibits a high anchor footprint, look for an adjacent cluster window
                if token_score >= 25.0:
                    cluster_tokens = [token]
                    
                    # Lookback phase: Collect adjacent tokens to the left on the same row
                    left_index = i - 1
                    while left_index >= 0:
                        left_token = line[left_index]
                        # Check spatial distance gap boundaries between neighbors
                        gap = cluster_tokens[0].bbox.top_left[0] - left_token.bbox.top_right[0]
                        
                        # Stop collecting if the gap exceeds normal word spacing boundaries (120px)
                        if 0 <= gap <= 120 or self._score_token_semantics(left_token.text) > 0:
                            cluster_tokens.insert(0, left_token)
                            left_index -= 1
                        else:
                            break
                            
                    # Lookahead phase: Collect adjacent tokens to the right on the same row
                    right_index = i + 1
                    while right_index < n:
                        right_token = line[right_index]
                        gap = right_token.bbox.top_left[0] - cluster_tokens[-1].bbox.top_right[0]
                        
                        if 0 <= gap <= 120 or self._score_token_semantics(right_token.text) > 0:
                            cluster_tokens.append(right_token)
                            right_index += 1
                            i = right_index - 1 # Advance pointer to avoid duplicate scanning loops
                        else:
                            break
                            
                    # -----------------------------------------------------------------
                    # CANDIDATE VALIDATION & ASSEMBLY FILTER
                    # -----------------------------------------------------------------
                    joined_text = " ".join([t.text for t in cluster_tokens])
                    
                    # Evaluate the collective metrics of the assembled candidate cluster
                    has_proto = any(t.text.lower() in self.protocols for t in cluster_tokens) or "http" in joined_text.lower()
                    has_tld = any(t.text.lower() in self.tlds for t in cluster_tokens) or any(f".{ext}" in joined_text.lower() for ext in self.tlds)
                    total_anchor_score = sum(self._score_token_semantics(t.text) for t in cluster_tokens)
                    
                    # Accept the cluster if it contains at least one strong anchor or multiple medium signals
                    if has_proto or has_tld or total_anchor_score >= 50.0:
                        candidates.append(URLCandidate(
                            tokens=cluster_tokens,
                            raw_joined_text=joined_text,
                            base_anchor_score=total_anchor_score,
                            contains_protocol=has_proto,
                            contains_tld=has_tld,
                            metadata={"line_index": line_index, "token_count": len(cluster_tokens)}
                        ))
                        
                i += 1
                
        logger.info(f"[✓] Candidate Detection Pass Complete. Isolated {len(candidates)} high-probability URL strings.")
        return candidates