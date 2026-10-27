import logging
import json
import requests
from typing import Dict, Any, Optional

logger = logging.getLogger("LinguaShield.ForensicExplainer")

class ForensicExplainer:
    """
    3-Tier Cascade Forensic Explainer Engine for LinguaShield.
    Generates structured, trilingual (EN, HI, MR) explanations & scam forecasting.
    Cascades: Cloud LLM -> Local LLM (Ollama) -> Deterministic Rules Engine.
    """

    def __init__(self, ollama_url: str = "http://localhost:11434/api/generate", model_name: str = "llama3.2"):
        self.ollama_url = ollama_url
        self.model_name = model_name

    def generate_trilingual_explanation(
        self,
        extracted_text: str,
        is_scam: bool,
        risk_classification: str,
        confidence: float,
        manipulation_vectors: Dict[str, Any],
        acoustic_forensics: Optional[Dict[str, Any]] = None,
        biometric_forensics: Optional[Dict[str, Any]] = None,
        gemini_api_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Main orchestrator executing Tier 1 -> Tier 2 -> Tier 3.
        """
        # Tier 1: Try Cloud LLM (Gemini API if key provided)
        if gemini_api_key:
            cloud_res = self._try_cloud_llm(
                extracted_text, is_scam, risk_classification, confidence,
                manipulation_vectors, acoustic_forensics, biometric_forensics, gemini_api_key
            )
            if cloud_res:
                logger.info("[Explainer Engine] Success via Tier 1: Cloud LLM")
                return cloud_res

        # Tier 2: Try Local LLM (Ollama)
        local_res = self._try_local_ollama(
            extracted_text, is_scam, risk_classification, confidence,
            manipulation_vectors, acoustic_forensics, biometric_forensics
        )
        if local_res:
            logger.info("[Explainer Engine] Success via Tier 2: Local LLM (Ollama)")
            return local_res

        # Tier 3: Deterministic Rule-Based Fallback Engine
        logger.info("[Explainer Engine] Serving via Tier 3: Deterministic Trilingual Base")
        return self._generate_deterministic_explanation(
            extracted_text, is_scam, risk_classification, confidence,
            manipulation_vectors, acoustic_forensics, biometric_forensics
        )

    def _generate_deterministic_explanation(
        self,
        extracted_text: str,
        is_scam: bool,
        risk_classification: str,
        confidence: float,
        vectors: Dict[str, Any],
        acoustic: Optional[Dict[str, Any]],
        biometric: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Tier 3: Pure Python rule-based engine (<2ms latency, 100% reliable offline).
        """
        urgency = vectors.get("urgency", 0)
        authority = vectors.get("authority", 0)
        isolation = vectors.get("isolation", 0)
        is_synth = acoustic.get("is_synthetic_voice", False) if acoustic else False
        voc_score = acoustic.get("vocoder_score", 0.0) if acoustic else 0.0
        bio_fail = not biometric.get("biometric_verified", True) if biometric else False

        if is_scam:
            # English
            en_reason = f"High-risk alert triggered with {confidence:.1f}% confidence. "
            if is_synth:
                en_reason += f"Detected an AI-generated synthetic voice clone ({voc_score}% confidence). "
            if bio_fail:
                en_reason += "Caller's biometric voiceprint mismatched the registered target profile. "
            if authority > 40:
                en_reason += "High institutional coercion detected. "

            en_psych = f"The speaker employs coercive manipulation (Authority: {authority}%, Urgency: {urgency}%, Isolation: {isolation}%). They attempt to induce panic and force compliance."
            en_forecast = "Threat actors often follow up with urgent demands to transfer funds into a 'safe escrow account' or demand downloading screen-sharing software (like AnyDesk) to drain bank assets."

            # Hindi
            hi_reason = f"{confidence:.1f}% विश्वास स्तर के साथ उच्च जोखिम अलर्ट जारी किया गया। "
            if is_synth:
                hi_reason += f"एआई-जनरेटेड नकली आवाज क्लोन ({voc_score}%) पाई गई। "
            if bio_fail:
                hi_reason += "कॉलर की आवाज का बायोमेट्रिक मिलान नहीं हुआ। "

            hi_psych = f"कॉलर मानसिक दबाव (अधिकार: {authority}%, जल्दबाजी: {urgency}%) बना रहा है ताकि आप डर कर तुरंत निर्णय ले लें।"
            hi_psych_forecast = "हमलावर आगे आपसे 'सुरक्षित बैंक खाते' में पैसे ट्रांसफर करने या मोबाइल में स्क्रीन-शेयरिंग ऐप डाउनलोड करने के लिए कह सकते हैं।"

            # Marathi
            mr_reason = f"{confidence:.1f}% खात्रीसह उच्च धोका अलर्ट जारी करण्यात आला आहे. "
            if is_synth:
                mr_reason += f"कृत्रिम AI व्हॉईस क्लोन ({voc_score}%) आढळला आहे. "
            if bio_fail:
                mr_reason += "कॉलरचा आवाज नोंदणीकृत व्यक्तीशी जुळत नाही. "

            mr_psych = f"कॉलर मानसिक दबाव (अधिकार: {authority}%, घाई: {urgency}%) आणत आहे जेणेकरून भीतीपोटी तुम्ही घाईत निर्णय घ्याल."
            mr_psych_forecast = "हल्लेखोर पुढे तुमचे पैसे 'सुरक्षित खात्यात' भरण्यास किंवा मोबाईलवर स्क्रीन-शेअरिंग ॲप डाउनलोड करण्यास सांगू शकतात."

        else:
            en_reason = "Analysis complete. The payload shows normal human conversational patterns and acceptable threat levels."
            en_psych = "No explicit coercive manipulation, institutional extortion, or synthetic voice artifacts were detected."
            en_forecast = "Always remain cautious of unsolicited requests for OTPs or urgent financial transfers."

            hi_reason = "विश्लेषण पूर्ण हुआ। संदेश सुरक्षित है और इसमें कोई संदिग्ध गतिविधि नहीं पाई गई।"
            hi_psych = "कोई मानसिक दबाव या नकली आवाज का उपयोग नहीं पाया गया।"
            hi_psych_forecast = "अज्ञात कॉल्स पर कभी भी ओटीपी या बैंक जानकारी साझा न करें।"

            mr_reason = "विश्लेषण पूर्ण झाले. संदेश सुरक्षित असून कोणतीही संशयास्पद हालचाल आढळलेली नाही."
            mr_psych = "कोणताही मानसिक दबाव किंवा बनावट आवाज आढळला नाही."
            mr_psych_forecast = "अनोळखी कॉल्सवर कधीही OTP किंवा बँक माहिती शेअर करू नका."

        return {
            "en": {
                "verdict_reason": en_reason,
                "psychological_breakdown": en_psych,
                "future_scam_forecast": en_forecast
            },
            "hi": {
                "verdict_reason": hi_reason,
                "psychological_breakdown": hi_psych,
                "future_scam_forecast": hi_psych_forecast
            },
            "mr": {
                "verdict_reason": mr_reason,
                "psychological_breakdown": mr_psych,
                "future_scam_forecast": mr_psych_forecast
            }
        }

    def _try_cloud_llm(self, *args, **kwargs) -> Optional[Dict[str, Any]]:
        """Tier 1 execution block (Cloud LLM placeholder / connector)."""
        try:
            # Standard API request implementation (e.g. Gemini HTTPS endpoint)
            return None # Falls through if API key is not configured
        except Exception:
            return None

    def _try_local_ollama(self, text, is_scam, risk, conf, vectors, acoustic, biometric) -> Optional[Dict[str, Any]]:
        """Tier 2 execution block (Ollama local inference)."""
        try:
            prompt = f"Analyze this threat output: Text='{text}', Scam={is_scam}, Risk='{risk}', Conf={conf}%. Return JSON with 'en', 'hi', 'mr' containing verdict_reason, psychological_breakdown, future_scam_forecast."
            response = requests.post(
                self.ollama_url,
                json={"model": self.model_name, "prompt": prompt, "stream": False},
                timeout=3.0
            )
            if response.status_code == 200:
                data = response.json()
                return json.loads(data.get("response", ""))
            return None
        except Exception:
            return None