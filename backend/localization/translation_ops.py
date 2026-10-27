import re
import threading
from typing import Optional, Tuple

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL_NAME = "facebook/nllb-200-distilled-600M"
TARGET_LANG = "eng_Latn"

_nllb_tokenizer = None
_nllb_model = None
_nllb_lock = threading.Lock()
_nllb_load_error: Optional[str] = None


# -------------------------------------------------------------------------
# EXISTING PHONETIC DIALECT MAP
# -------------------------------------------------------------------------
REGIONAL_PHONETIC_MAP = {
    "turant": "immediate",
    "jaldi": "quick",
    "abhee": "now",
    "abhi": "now",
    "aaj": "today",
    "akhri": "final",
    "ekdom": "urgent",
    "bachane": "save",
    "band": "blocked",
    "rok": "stopped",

    "paise": "money",
    "khata": "account",
    "rupaye": "rupees",
    "inami": "lottery",
    "jeeta": "won",
    "inaam": "prize",
    "shulk": "fees",
    "kar": "tax",

    "adhikari": "officer",
    "pulis": "police",
    "thana": "police station",
    "daroga": "inspector",
    "adalat": "court",
    "sarkari": "government",
    "giraftari": "arrest",

    "chupchap": "secret",
    "gopniya": "confidential",
    "kisi ko mat batana": "don't tell anyone",
    "mat bolo": "don't speak",
    "line par raho": "stay online",
}


# -------------------------------------------------------------------------
# SECURITY-SENSITIVE ENTITY PROTECTION
# -------------------------------------------------------------------------
_ENTITY_PATTERNS = [
    r"https?://[^\s]+",
    r"\b(?:www\.)?[A-Za-z0-9.-]+\.(?:com|net|org|gov|in|co|io|ai|xyz)(?:/[^\s]*)?",
    r"(?:₹|Rs\.?|INR)\s*[\d,]+(?:\.\d+)?",
    r"\b\d[\d,]{2,}\b",
    r"\bOTP\b",
    r"\bPIN\b",
    r"\bCVV\b",
    r"\bUPI\b",
    r"\bNEFT\b",
    r"\bRTGS\b",
    r"\bIMPS\b",
    r"\bIFSC\b",
    r"\bAnyDesk\b",
    r"\bTeamViewer\b",
]


def _protect_security_entities(text: str) -> Tuple[str, dict]:
    protected = text
    entity_map = {}
    counter = 0

    for pattern in _ENTITY_PATTERNS:
        regex = re.compile(pattern, flags=re.IGNORECASE)

        def repl(match):
            nonlocal counter

            original = match.group(0)
            placeholder = f"LSENTITY{counter}TOKEN"
            counter += 1

            entity_map[placeholder] = original
            return placeholder

        protected = regex.sub(repl, protected)

    return protected, entity_map


def _restore_security_entities(text: str, entity_map: dict) -> str:
    restored = text

    for placeholder, original in entity_map.items():
        restored = restored.replace(placeholder, original)

    return restored


# -------------------------------------------------------------------------
# SCRIPT / LANGUAGE DETECTION
# -------------------------------------------------------------------------
DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")


MARATHI_MARKERS = (
    "तुमचा",
    "तुमची",
    "तुमचे",
    "तुमच्या",
    "आहे",
    "आहेत",
    "होईल",
    "झाले",
    "झालेल्या",
    "पाठवा",
    "करा",
    "करू नका",
    "शेअर करू नका",
    "कोणासोबत",
    "आत्ताच",
    "कृपया",
    "यासाठी",
    "याबद्दल",
    "खाते",
    "खात्यात",
    "पडताळणी",
    "सूचना",
    "विभागातून",
    "विभागाचा",
    "विनंती",
    "तातडीची",
    "गुंतवणूक",
    "फसवणूक",
    "पैसे",
    "पाठव",
    "मागणार",
    "नाहीत",
    "नको",
    "तुम्हाला",
    "तुम्ही",
    "माझा",
    "माझी",
    "माझे",
)


HINDI_MARKERS = (
    "आपका",
    "आपकी",
    "आपके",
    "अपना",
    "अपने",
    "है",
    "हैं",
    "करें",
    "भेजें",
    "भेजना",
    "न करें",
    "अभी",
    "तुरंत",
    "किसी",
    "किसी को",
    "खाता",
    "खाते",
    "खातों",
    "पैसा",
    "पैसे",
    "चाहिए",
    "कृपया",
    "सूचना",
    "विभाग",
    "सत्यापन",
    "प्रमाणित",
    "भुगतान",
    "धोखाधड़ी",
    "नहीं",
    "आपको",
    "आप",
    "मुझे",
    "हमारा",
    "हमारे",
    "करना",
    "करने",
    "जाएं",
    "जाएँ",
)


def _count_marker_hits(text: str, markers: tuple) -> int:
    lowered = text.lower()

    return sum(
        1
        for marker in markers
        if marker.lower() in lowered
    )


def detect_indic_language(
    text: str,
    source_lang: str = "auto",
) -> Optional[str]:

    aliases = {
        "hi": "hin_Deva",
        "hin": "hin_Deva",
        "hindi": "hin_Deva",
        "hin_deva": "hin_Deva",

        "mr": "mar_Deva",
        "mar": "mar_Deva",
        "marathi": "mar_Deva",
        "mar_deva": "mar_Deva",
    }

    if source_lang:
        source_key = source_lang.lower().strip()

        if source_key in aliases:
            return aliases[source_key]

    if not DEVANAGARI_RE.search(text):
        return None

    marathi_score = _count_marker_hits(
        text,
        MARATHI_MARKERS,
    )

    hindi_score = _count_marker_hits(
        text,
        HINDI_MARKERS,
    )

    if marathi_score > hindi_score and marathi_score >= 1:
        return "mar_Deva"

    return "hin_Deva"


# -------------------------------------------------------------------------
# NLLB MODEL LOADER
# -------------------------------------------------------------------------
def _load_nllb():
    global _nllb_tokenizer
    global _nllb_model
    global _nllb_load_error

    if (
        _nllb_model is not None
        and _nllb_tokenizer is not None
    ):
        return _nllb_tokenizer, _nllb_model

    with _nllb_lock:

        if (
            _nllb_model is not None
            and _nllb_tokenizer is not None
        ):
            return _nllb_tokenizer, _nllb_model

        if _nllb_load_error is not None:
            raise RuntimeError(_nllb_load_error)

        try:
            print(
                f"[Localization] Loading NLLB model: "
                f"{MODEL_NAME}"
            )

            tokenizer = AutoTokenizer.from_pretrained(
                MODEL_NAME
            )

            model = AutoModelForSeq2SeqLM.from_pretrained(
                MODEL_NAME
            )

            device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

            model = model.to(device)
            model.eval()

            _nllb_tokenizer = tokenizer
            _nllb_model = model

            print(
                f"[Localization] NLLB ready on "
                f"{device.type}: {MODEL_NAME}"
            )

            return (
                _nllb_tokenizer,
                _nllb_model,
            )

        except Exception as exc:

            _nllb_load_error = (
                "NLLB initialization failed: "
                f"{type(exc).__name__}: {exc}"
            )

            print(
                f"[Localization] ERROR: "
                f"{_nllb_load_error}"
            )

            raise RuntimeError(
                _nllb_load_error
            ) from exc


# -------------------------------------------------------------------------
# DIGITAL ARREST DEFENSIVE TRANSLATION NORMALIZATION
# -------------------------------------------------------------------------
#
# Targeted fix only.
#
# A safety advisory can legitimately contain:
#
#   police
#   digital arrest
#   transfer money
#   arrest
#
# But the sentence may explicitly say:
#
#   "Real police officers do not ask people to transfer money."
#
# That is defensive content, not a malicious instruction.
#
# This function identifies those clearly defensive constructions and adds
# an explicit defensive statement for the downstream English context layer.
# It does NOT modify affirmative requests.
# -------------------------------------------------------------------------
def _normalize_digital_arrest_defensive_translation(
    text: str,
) -> str:

    if not text:
        return text

    normalized = str(text)

    # ------------------------------------------------------------------
    # 1. Educational / awareness indicators
    # ------------------------------------------------------------------
    educational_terms = [
        "digital arrest is a fraud",
        "digital arrest is a scam",
        "digital arrest scams",
        "digital arrest fraud",
        "cyber security awareness",
        "cybersecurity awareness",
        "security awareness",
        "security advisory",
        "safety advisory",
        "fraud awareness",
        "scam awareness",
    ]

    # ------------------------------------------------------------------
    # 2. Defensive police / law-enforcement statements
    # ------------------------------------------------------------------
    defensive_police_patterns = [
        (
            r"\b(?:real|genuine|legitimate)\s+"
            r"police\s+officers?\b.*?"
            r"\b(?:do not|don't|never)\b.*?"
            r"\b(?:digital arrest|arrest)\b"
        ),

        (
            r"\b(?:real|genuine|legitimate)\s+"
            r"police\s+officers?\b.*?"
            r"\b(?:do not|don't|never)\b.*?"
            r"\b(?:ask|tell|require|demand)\b.*?"
            r"\b(?:transfer|send|pay|deposit)\b.*?"
            r"\b(?:money|funds|cash)\b"
        ),

        (
            r"\bpolice\s+officers?\b.*?"
            r"\b(?:do not|don't|never)\b.*?"
            r"\b(?:ask|tell|require|demand)\b.*?"
            r"\b(?:transfer|send|pay|deposit)\b.*?"
            r"\b(?:money|funds|cash)\b"
        ),
    ]

    has_defensive_police_statement = any(
        re.search(
            pattern,
            normalized,
            flags=re.IGNORECASE | re.DOTALL,
        )
        for pattern in defensive_police_patterns
    )

    # ------------------------------------------------------------------
    # 3. Explicit defensive money statements
    # ------------------------------------------------------------------
    defensive_money_patterns = [
        (
            r"\b(?:do not|don't|never)\b.{0,120}"
            r"\b(?:transfer|send|pay|deposit)\b.{0,80}"
            r"\b(?:money|funds|cash)\b"
        ),

        (
            r"\b(?:do not|don't|never)\b.{0,120}"
            r"\b(?:ask|tell|require|demand)\b.{0,100}"
            r"\b(?:transfer|send|pay|deposit)\b.{0,80}"
            r"\b(?:money|funds|cash)\b"
        ),
    ]

    has_defensive_money_statement = any(
        re.search(
            pattern,
            normalized,
            flags=re.IGNORECASE | re.DOTALL,
        )
        for pattern in defensive_money_patterns
    )

    # ------------------------------------------------------------------
    # 4. Safe-response / verification language
    # ------------------------------------------------------------------
    safe_response_patterns = [
        (
            r"\b(?:end|stop|ignore|disconnect|report)\b.*?"
            r"\b(?:call|message|conversation|request)\b"
        ),

        (
            r"\b(?:contact|call|verify)\b.*?"
            r"\b(?:official|independent|organization|institution)\b"
        ),

        (
            r"\bdo not\b.*?"
            r"\b(?:contact|send|transfer|pay|provide)\b"
        ),
    ]

    has_safe_response_language = any(
        re.search(
            pattern,
            normalized,
            flags=re.IGNORECASE | re.DOTALL,
        )
        for pattern in safe_response_patterns
    )

    # ------------------------------------------------------------------
    # 5. Determine whether the translated text is an advisory.
    # ------------------------------------------------------------------
    lowered = normalized.lower()

    has_digital_arrest_reference = (
        "digital arrest" in lowered
        or "arrest" in lowered
    )

    has_educational_language = any(
        term in lowered
        for term in educational_terms
    )

    is_digital_arrest_advisory = (
        (
            has_educational_language
            and has_digital_arrest_reference
        )
        or has_defensive_police_statement
        or (
            has_digital_arrest_reference
            and has_defensive_money_statement
        )
    )

    if is_digital_arrest_advisory:

        return (
            normalized
            + " "
            "SECURITY AWARENESS: "
            "This is a warning about a digital arrest scam. "
            "Do not transfer money based on a supposed police request. "
            "Do not comply with the request. "
            "Verify the situation independently through official channels."
        )

    # ------------------------------------------------------------------
    # 6. General defensive money advisory.
    # ------------------------------------------------------------------
    if (
        has_defensive_money_statement
        and has_safe_response_language
    ):

        return (
            normalized
            + " "
            "SECURITY AWARENESS: "
            "Do not transfer money. "
            "Verify the request independently through official channels."
        )

    return normalized


# -------------------------------------------------------------------------
# NLLB TRANSLATION
# -------------------------------------------------------------------------
def _translate_native_script_to_english(
    text: str,
    source_lang: str,
) -> str:

    tokenizer, model = _load_nllb()

    protected_text, entity_map = (
        _protect_security_entities(text)
    )

    tokenizer.src_lang = source_lang

    device = next(
        model.parameters()
    ).device

    encoded = tokenizer(
        protected_text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    encoded = {
        key: value.to(device)
        for key, value in encoded.items()
    }

    target_lang_id = (
        tokenizer.convert_tokens_to_ids(
            TARGET_LANG
        )
    )

    with torch.no_grad():

        generated = model.generate(
            **encoded,
            forced_bos_token_id=target_lang_id,
            max_length=512,
            num_beams=5,
            early_stopping=True,
        )

    translated = tokenizer.batch_decode(
        generated,
        skip_special_tokens=True,
    )[0].strip()

    translated = _restore_security_entities(
        translated,
        entity_map,
    )

    # Targeted Digital Arrest defensive-context fix.
    translated = (
        _normalize_digital_arrest_defensive_translation(
            translated
        )
    )

    return translated or text


# -------------------------------------------------------------------------
# LATIN-SCRIPT PHONETIC NORMALIZATION
# -------------------------------------------------------------------------
def _normalize_phonetic_latin(
    text: str,
) -> str:

    normalized_text = text.lower().strip()

    for (
        regional_word,
        english_word,
    ) in REGIONAL_PHONETIC_MAP.items():

        pattern = re.compile(
            rf"(?<!\w)"
            rf"{re.escape(regional_word)}"
            rf"(?!\w)",
            re.IGNORECASE,
        )

        normalized_text = pattern.sub(
            english_word,
            normalized_text,
        )

    return normalized_text


# -------------------------------------------------------------------------
# PUBLIC TRANSLATION API
# -------------------------------------------------------------------------
def translate_content_to_english(
    text: str,
    source_lang: str = "auto",
) -> str:

    if not isinstance(text, str):
        return ""

    original_text = text.strip()

    if not original_text:
        return ""

    normalized_input = re.sub(
        r"\s+",
        " ",
        original_text,
    )

    native_lang = detect_indic_language(
        normalized_input,
        source_lang=source_lang,
    )

    # Native Hindi / Marathi route.
    if native_lang in {
        "hin_Deva",
        "mar_Deva",
    }:

        print(
            "[Localization] Native Devanagari detected. "
            f"Using NLLB source={native_lang}."
        )

        try:

            translated = (
                _translate_native_script_to_english(
                    normalized_input,
                    native_lang,
                )
            )

            print(
                "[Localization] Translation completed "
                f"successfully: {native_lang} -> "
                f"{TARGET_LANG}"
            )

            print(
                "[Localization] English analysis text: "
                f"{translated}"
            )

            return translated

        except Exception as exc:

            # IMPORTANT:
            # Never manufacture a threat when translation fails.
            print(
                "[Localization] WARNING: Native-script "
                "translation failed. Falling back to "
                f"original text. Reason: {exc}"
            )

            return normalized_input

    # Existing Latin / Hinglish / Marathlish route.
    print(
        "[Localization] Latin/mixed-Latin input detected. "
        "Processing phonetic normalization pass..."
    )

    processed_output = _normalize_phonetic_latin(
        normalized_input
    )

    print(
        "[Localization] Normalization completed successfully."
    )

    return processed_output


# -------------------------------------------------------------------------
# OPTIONAL HEALTH DIAGNOSTIC
# -------------------------------------------------------------------------
def localization_health() -> dict:

    return {
        "model": MODEL_NAME,
        "target_language": TARGET_LANG,
        "nllb_loaded": (
            _nllb_model is not None
        ),
        "nllb_load_error": _nllb_load_error,
        "torch_version": torch.__version__,
        "device_available": (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        ),
    }


# -------------------------------------------------------------------------
# OPTIONAL EXPLICIT MODEL WARM-UP
# -------------------------------------------------------------------------
def warmup_translation_model() -> bool:

    try:

        _load_nllb()
        return True

    except Exception as exc:

        print(
            f"[Localization] Warm-up failed: {exc}"
        )

        return False