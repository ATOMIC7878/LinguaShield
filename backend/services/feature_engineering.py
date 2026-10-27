import re
import logging

logger = logging.getLogger("LinguaShield.FeatureEngineering")

class FinancialFraudEvaluator:
    """
    Analyzes cross-script transcripts to calculate the Financial Fraud Lure Vector.
    Maps out the semantic convergence of Asset Lures, Returns Multipliers, and Upfront Fees.
    """
    def __init__(self):
        # 1. Financial Asset Lures (English, Devanagari, and common Transliterations)
        self.financial_lures = re.compile(
            r'(stock|market|invest|money|cash|paisa|wealth|shares|trading|'
            r'स्टॉक|मार्केट|इन्वेस्ट|इनवेस्टमेंट|पैसा|पैसे|शेयर|ट्रेडिंग)', 
            re.IGNORECASE
        )
        
        # 2. High-Yield Return Multipliers & Certainty Hooks
        self.return_multipliers = re.compile(
            r'(double|triple|guaranteed|returns|profit|bonus|multiply|badwa|badhava|'
            r'डबल|डबल|गारंटी|गॅरंटी|मुनाफा|प्रॉफिट|बड़वा|बढ़वा|रिटर्न)', 
            re.IGNORECASE
        )
        
        # 3. Upfront Friction / Immediate Cost Demands
        self.upfront_friction = re.compile(
            r'(initial|payment|pay|deposit|fee|charge|transfer|advance|'
            r'इनिशियल|पेमेंट|पे|डिपॉजिट|फीस|चार्ज|एडवांस|ट्रांसफर)', 
            re.IGNORECASE
        )

    def extract_lure_metrics(self, transcript: str) -> dict:
        """
        Ingests the complete transcript matrix and calculates sub-vector signals
        alongside a normalized, compound Financial Fraud Lure Score.
        """
        if not transcript.strip():
            return {
                "financial_asset_intensity": 0.0,
                "multiplier_certainty_intensity": 0.0,
                "upfront_friction_intensity": 0.0,
                "compound_financial_lure_score": 0.0
            }

        # Normalize text variations to safe matching bounds
        normalized_text = transcript.lower().strip()
        total_word_count = max(1, len(normalized_text.split()))

        # Calculate exact hit counts across independent linguistic dimensions
        financial_hits = len(self.financial_lures.findall(normalized_text))
        multiplier_hits = len(self.return_multipliers.findall(normalized_text))
        friction_hits = len(self.upfront_friction.findall(normalized_text))

        # Calculate localized feature intensities relative to document scale
        financial_intensity = min(1.0, financial_hits / 3.0)  # Caps max saturation at 3 key hits
        multiplier_intensity = min(1.0, multiplier_hits / 2.0) # Caps max saturation at 2 key hits
        friction_intensity = min(1.0, friction_hits / 2.0)   # Caps max saturation at 2 key hits

        # ─── THE SEMANTIC CONVERGENCE LAW ───
        # A true financial scam requires all three components to converge: 
        # Asset + Multiplier Hint + Upfront Cost Demand.
        # If any component is completely absent (0.0), the compound risk floor drops heavily.
        if financial_hits > 0 and multiplier_hits > 0 and friction_hits > 0:
            # High-intensity convergence path
            compound_score = (financial_intensity * 0.3) + (multiplier_intensity * 0.4) + (friction_intensity * 0.3)
            # Apply a convergence safety boost to ensure it breaks past Random Forest safe limits
            compound_score = min(1.0, compound_score * 1.2)
        else:
            # Linear decay path if components are disjointed
            compound_score = (financial_intensity * 0.2) + (multiplier_intensity * 0.2) + (friction_intensity * 0.1)

        metrics_payload = {
            "financial_asset_intensity": round(financial_intensity * 100, 2),
            "multiplier_certainty_intensity": round(multiplier_intensity * 100, 2),
            "upfront_friction_intensity": round(friction_intensity * 100, 2),
            "compound_financial_lure_score": round(compound_score * 100, 2)
        }

        logger.info(f"[Feature Extraction Complete] Calculated Matrix Payload: {metrics_payload}")
        return metrics_payload