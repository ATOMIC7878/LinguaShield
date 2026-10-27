"""
LinguaShield - TrustPath Adapter
--------------------------------

Purpose:
    Convert existing LinguaShield analysis results into a normalized
    TrustPath evidence format.

Architecture:

    Existing module output
            ↓
        TrustPathAdapter
            ↓
       normalized evidence
            ↓
      TrustPath orchestrator

IMPORTANT:
    This adapter must not modify or call the internal implementation of
    the existing detection modules.

    The adapter only reads their result objects/dictionaries and converts
    whatever information is available into the TrustPath contract.

Location:
    backend/services/trustpath/adapter.py
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Iterable, List, Optional


# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

SUPPORTED_SOURCES = {
    "text",
    "visual",
    "acoustic",
    "biometric",
}


# ---------------------------------------------------------------------------
# BASIC HELPERS
# ---------------------------------------------------------------------------

def _to_dict(value: Any) -> Dict[str, Any]:
    """
    Convert a supported result object into a dictionary.

    Supports:
        - dict
        - Pydantic models exposing model_dump()
        - objects exposing dict()
        - plain Python objects exposing __dict__

    Unknown/unconvertible values become an empty dictionary.
    """

    if value is None:
        return {}

    if isinstance(value, dict):
        return deepcopy(value)

    model_dump = getattr(value, "model_dump", None)

    if callable(model_dump):
        try:
            result = model_dump()
            if isinstance(result, dict):
                return deepcopy(result)
        except Exception:
            pass

    object_dict = getattr(value, "dict", None)

    if callable(object_dict):
        try:
            result = object_dict()
            if isinstance(result, dict):
                return deepcopy(result)
        except Exception:
            pass

    raw_dict = getattr(value, "__dict__", None)

    if isinstance(raw_dict, dict):
        return deepcopy(raw_dict)

    return {}


def _first_present(
    data: Dict[str, Any],
    keys: Iterable[str],
) -> Any:
    """
    Return the first non-None field from the supplied keys.
    """

    for key in keys:
        if key in data and data[key] is not None:
            return data[key]

    return None


def _string_or_none(value: Any) -> Optional[str]:
    """
    Convert a value to a clean string where appropriate.
    """

    if value is None:
        return None

    if isinstance(value, str):
        value = value.strip()
        return value or None

    return str(value)


def _float_or_none(value: Any) -> Optional[float]:
    """
    Safely convert a numeric confidence-like value.

    Values between 0 and 1:
        assumed normalized.

    Values between 0 and 100:
        converted to 0..1.

    Other values:
        ignored.
    """

    if value is None:
        return None

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None

    if 0.0 <= numeric <= 1.0:
        return numeric

    if 0.0 <= numeric <= 100.0:
        return numeric / 100.0

    return None


def _new_evidence_id(source: str, index: int) -> str:
    """
    Create a deterministic ID for the current adapter pass.

    The final case-level ID is assigned by TrustPath.
    """

    return f"adapter_{source}_{index}"


# ---------------------------------------------------------------------------
# RESULT UNWRAPPING
# ---------------------------------------------------------------------------

def _unwrap_result(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Safely unwrap common result-container structures.

    Some existing modules may return:

        {
            "result": {...}
        }

    or:

        {
            "analysis": {...}
        }

    or:

        {
            "data": {...}
        }

    TrustPath should accept these structures without requiring changes
    to the existing modules.

    Important:
        The original object is preserved whenever possible.
    """

    if not isinstance(data, dict):
        return {}

    current = deepcopy(data)

    wrapper_keys = (
        "result",
        "analysis",
        "output",
        "prediction",
        "response",
        "data",
    )

    # Only unwrap when the wrapper contains a dictionary.
    # Limit depth to avoid accidentally traversing arbitrary user data.
    for _ in range(3):
        found = False

        for key in wrapper_keys:
            nested = current.get(key)

            if isinstance(nested, dict):
                current = nested
                found = True
                break

        if not found:
            break

    return current


def _merged_result_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Preserve the original result while also exposing normalized fields
    from a common nested result container.

    Nested normalized fields are copied to the top level only when a
    top-level field does not already exist.
    """

    if not isinstance(data, dict):
        return {}

    original = deepcopy(data)
    normalized = _unwrap_result(data)

    if normalized is original:
        return original

    merged = deepcopy(original)

    for key, value in normalized.items():
        if key not in merged:
            merged[key] = deepcopy(value)

    return merged


# ---------------------------------------------------------------------------
# FIELD NORMALIZATION
# ---------------------------------------------------------------------------

def _normalize_common_identity_fields(
    data: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Normalize explicit identity/entity fields into the names expected by
    the TrustPath entity extractor.

    This does NOT infer identity.

    It only maps fields that already exist in upstream output.
    """

    normalized = deepcopy(data)

    # Person
    person_name = _first_present(
        normalized,
        (
            "person_name",
            "sender_name",
            "recipient_name",
            "person",
            "name",
            "speaker_name",
            "contact_name",
        ),
    )

    if person_name is not None:
        normalized.setdefault("person_name", person_name)

    # Organization
    organization_name = _first_present(
        normalized,
        (
            "organization_name",
            "company_name",
            "organization",
            "company",
        ),
    )

    if organization_name is not None:
        normalized.setdefault("organization_name", organization_name)

    # Brand
    brand_name = _first_present(
        normalized,
        (
            "brand_name",
            "brand",
            "detected_brand",
            "logo_brand",
        ),
    )

    if brand_name is not None:
        normalized.setdefault("brand_name", brand_name)

    # Phone
    phone = _first_present(
        normalized,
        (
            "phone_number",
            "phone",
            "mobile",
            "contact_number",
        ),
    )

    if phone is not None:
        normalized.setdefault("phone_number", phone)

    # Email
    email = _first_present(
        normalized,
        (
            "email_address",
            "email",
            "mail",
            "contact_email",
        ),
    )

    if email is not None:
        normalized.setdefault("email_address", email)

    # Transaction
    transaction_id = _first_present(
        normalized,
        (
            "transaction_id",
            "transaction",
            "txn_id",
            "transaction_reference",
            "transaction_number",
        ),
    )

    if transaction_id is not None:
        normalized.setdefault("transaction_id", transaction_id)

    # Invoice
    invoice_number = _first_present(
        normalized,
        (
            "invoice_number",
            "invoice_id",
            "invoice",
            "bill_number",
            "invoice_reference",
        ),
    )

    if invoice_number is not None:
        normalized.setdefault("invoice_number", invoice_number)

    # Product
    product_name = _first_present(
        normalized,
        (
            "product_name",
            "product",
            "item_name",
        ),
    )

    if product_name is not None:
        normalized.setdefault("product_name", product_name)

    # URL
    url = _first_present(
        normalized,
        (
            "url",
            "link",
            "website",
            "website_url",
        ),
    )

    if url is not None:
        normalized.setdefault("url", url)

    return normalized


def _normalize_relationship_fields(
    data: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Normalize explicit relationship metadata.

    No relationship is inferred from a name alone.
    """

    normalized = deepcopy(data)

    sender_entity_id = _first_present(
        normalized,
        (
            "sender_entity_id",
            "subject_entity_id",
        ),
    )

    recipient_entity_id = _first_present(
        normalized,
        (
            "recipient_entity_id",
            "object_entity_id",
        ),
    )

    relationship = _first_present(
        normalized,
        (
            "relationship",
            "relation",
            "relationship_type",
        ),
    )

    if sender_entity_id is not None:
        normalized.setdefault(
            "sender_entity_id",
            sender_entity_id,
        )

    if recipient_entity_id is not None:
        normalized.setdefault(
            "recipient_entity_id",
            recipient_entity_id,
        )

    if relationship is not None:
        normalized.setdefault(
            "relationship",
            relationship,
        )

    return normalized


def _normalize_visual_fields(
    data: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Normalize common Visual fields while preserving all original fields.
    """

    normalized = _add_common_scenario_signals(deepcopy(data))

    classification = _first_present(
        normalized,
        (
            "visual_classification",
            "classification",
            "label",
            "category",
            "threat_category",
        ),
    )

    if classification is not None:
        normalized.setdefault(
            "visual_classification",
            classification,
        )

    ocr = _first_present(
        normalized,
        (
            "ocr",
            "ocr_text",
            "extracted_text",
            "text",
        ),
    )

    if ocr is not None:
        normalized.setdefault("ocr", ocr)

    objects = _first_present(
        normalized,
        (
            "detected_objects",
            "objects",
            "object_detection",
        ),
    )

    if objects is not None:
        normalized.setdefault(
            "detected_objects",
            objects,
        )

    brands = _first_present(
        normalized,
        (
            "detected_brands",
            "brands",
            "brand",
            "logo",
        ),
    )

    if brands is not None:
        normalized.setdefault(
            "detected_brands",
            brands,
        )

    return normalized


def _scenario_signal_from_risk(risk_classification: Any) -> Optional[str]:
    """Derive a stable scenario label from an explicit upstream classification."""
    if not isinstance(risk_classification, str):
        return None
    value = risk_classification.strip()
    if not value:
        return None
    if ":" in value:
        value = value.split(":", 1)[1].strip()
    value = value.lower()
    value = value.replace("/", " ").replace("&", " and ")
    tokens = [part for part in value.replace("-", " ").split() if part]
    return "_".join(tokens) or None


def _add_common_scenario_signals(data: Dict[str, Any]) -> Dict[str, Any]:
    """Map explicit upstream threat outputs into TrustPath scenario/event fields."""
    normalized = deepcopy(data)
    risk_classification = _first_present(
        normalized,
        ("risk_classification", "risk_class", "threat_classification"),
    )
    if risk_classification is not None:
        normalized.setdefault("risk_classification", risk_classification)
        scenario_type = _scenario_signal_from_risk(risk_classification)
        if scenario_type and str(risk_classification).upper() != "CLEARED CORE ANALYSIS":
            normalized.setdefault("scenario_type", scenario_type)
            normalized.setdefault("scenario_summary", str(risk_classification))

    confidence = _first_present(
        normalized,
        ("confidence", "score", "classification_confidence", "analysis_confidence"),
    )
    confidence_norm = _float_or_none(confidence)
    if confidence_norm is not None:
        normalized.setdefault("scenario_confidence", confidence_norm)

    is_scam = normalized.get("is_scam")
    if is_scam is not None:
        normalized.setdefault("threat_detected", bool(is_scam))

    vectors = normalized.get("manipulation_vectors")
    if isinstance(vectors, dict):
        normalized.setdefault("threat_indicators", vectors)

    if risk_classification is not None:
        normalized.setdefault("event_title", "Upstream threat analysis")
        normalized.setdefault("event_description", str(risk_classification))
        normalized.setdefault("event_type", "threat_classification")

    return normalized


def _normalize_text_fields(
    data: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Normalize common Text analysis fields.
    """

    normalized = _add_common_scenario_signals(deepcopy(data))

    classification = _first_present(
        normalized,
        (
            "classification",
            "label",
            "category",
            "threat_category",
            "risk_classification",
            "prediction",
        ),
    )

    if classification is not None:
        normalized.setdefault(
            "classification",
            classification,
        )

    intent = _first_present(
        normalized,
        (
            "intent",
            "detected_intent",
            "user_intent",
            "message_intent",
        ),
    )

    if intent is not None:
        normalized.setdefault(
            "intent",
            intent,
        )

    indicators = _first_present(
        normalized,
        (
            "threat_indicators",
            "indicators",
            "signals",
            "red_flags",
            "risk_indicators",
        ),
    )

    if indicators is not None:
        normalized.setdefault(
            "threat_indicators",
            indicators,
        )

    return normalized


def _normalize_acoustic_fields(
    data: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Normalize common Acoustic/deepfake fields.
    """

    normalized = _add_common_scenario_signals(deepcopy(data))

    nested = normalized.get("acoustic_forensics")
    if isinstance(nested, dict):
        for key, value in nested.items():
            normalized.setdefault(key, deepcopy(value))


    vocoder_score = _first_present(
        normalized,
        (
            "vocoder_score",
            "deepfake_score",
            "synthetic_score",
        ),
    )

    if vocoder_score is not None:
        normalized.setdefault(
            "vocoder_score",
            vocoder_score,
        )

    synthetic_detected = _first_present(
        normalized,
        (
            "synthetic_detected",
            "is_synthetic",
            "is_synthetic_voice",
            "synthetic",
        ),
    )

    if synthetic_detected is not None:
        normalized.setdefault(
            "synthetic_detected",
            synthetic_detected,
        )

    spectral_flatness = _first_present(
        normalized,
        (
            "spectral_flatness",
            "flatness",
        ),
    )

    if spectral_flatness is not None:
        normalized.setdefault(
            "spectral_flatness",
            spectral_flatness,
        )

    return normalized


def _normalize_biometric_fields(
    data: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Normalize common Biometric verification fields.
    """

    normalized = _add_common_scenario_signals(deepcopy(data))

    nested = normalized.get("biometric_forensics")
    if isinstance(nested, dict):
        for key, value in nested.items():
            normalized.setdefault(key, deepcopy(value))


    match = _first_present(
        normalized,
        (
            "match",
            "is_match",
            "verified",
            "biometric_verified",
            "identity_match",
        ),
    )

    if match is not None:
        normalized.setdefault(
            "match",
            match,
        )

    similarity = _first_present(
        normalized,
        (
            "similarity",
            "similarity_score",
            "match_score",
            "verification_score",
        ),
    )

    if similarity is not None:
        normalized.setdefault(
            "similarity",
            similarity,
        )

    threshold = _first_present(
        normalized,
        (
            "threshold",
            "similarity_threshold",
            "match_threshold",
            "similarity_threshold",
        ),
    )

    if threshold is not None:
        normalized.setdefault(
            "threshold",
            threshold,
        )

    return normalized


# ---------------------------------------------------------------------------
# NORMALIZED EVIDENCE FORMAT
# ---------------------------------------------------------------------------

def make_evidence(
    *,
    source: str,
    evidence_type: str,
    title: Optional[str] = None,
    description: Optional[str] = None,
    content: Optional[str] = None,
    user_context: Optional[str] = None,
    status: str = "detected",
    confidence: Optional[float] = None,
    detected_data: Optional[Dict[str, Any]] = None,
    source_reference: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create one normalized TrustPath evidence object.

    Mirrors the EvidenceItem contract used by TrustPath.
    """

    return {
        "evidence_id": None,
        "source": source,
        "evidence_type": evidence_type,
        "title": title,
        "description": description,
        "content": content,
        "user_context": user_context,
        "status": status,
        "confidence": confidence,
        "detected_data": deepcopy(detected_data or {}),
        "source_reference": source_reference,
        "related_entity_ids": [],
        "related_event_ids": [],
    }


# ---------------------------------------------------------------------------
# TEXT ADAPTER
# ---------------------------------------------------------------------------

def adapt_text_result(
    result: Any,
    *,
    user_context: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Adapt an existing Text analysis result.

    No Text model is called here.
    """

    raw_data = _to_dict(result)

    if not raw_data:
        return None

    data = _merged_result_data(raw_data)
    data = _normalize_text_fields(data)
    data = _normalize_common_identity_fields(data)
    data = _normalize_relationship_fields(data)

    confidence = _float_or_none(
        _first_present(
            data,
            (
                "confidence",
                "score",
                "classification_confidence",
                "analysis_confidence",
            ),
        )
    )

    detected_data = deepcopy(data)

    if user_context:
        detected_data.setdefault(
            "user_context",
            user_context,
        )

    return make_evidence(
        source="text",
        evidence_type="text_analysis",
        title="Text analysis",
        description="Existing Text analysis supplied to TrustPath.",
        content=_string_or_none(data.get("processed_text") or data.get("text")),
        user_context=user_context,
        status="detected",
        confidence=confidence,
        detected_data=detected_data,
        source_reference=_string_or_none(
            _first_present(
                data,
                (
                    "source_reference",
                    "message_id",
                    "request_id",
                    "id",
                ),
            )
        ),
    )


# ---------------------------------------------------------------------------
# VISUAL ADAPTER
# ---------------------------------------------------------------------------

def adapt_visual_result(
    result: Any,
    *,
    user_context: Optional[str] = None,
    visual_context: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Adapt an existing Visual analysis result.

    OCR, classification, object detection, brands and other source fields
    remain inside detected_data.
    """

    raw_data = _to_dict(result)

    if not raw_data:
        return None

    data = _merged_result_data(raw_data)
    data = _normalize_visual_fields(data)
    data = _normalize_common_identity_fields(data)
    data = _normalize_relationship_fields(data)

    confidence = _float_or_none(
        _first_present(
            data,
            (
                "confidence",
                "score",
                "classification_confidence",
                "analysis_confidence",
            ),
        )
    )

    detected_data = deepcopy(data)

    if visual_context:
        detected_data["visual_context"] = visual_context

    if user_context:
        detected_data.setdefault(
            "user_context",
            user_context,
        )

    return make_evidence(
        source="visual",
        evidence_type="visual_analysis",
        title="Visual analysis",
        description="Existing Visual analysis supplied to TrustPath.",
        content=_string_or_none(data.get("extracted_text") or data.get("ocr") or data.get("text")),
        user_context=user_context,
        status="detected",
        confidence=confidence,
        detected_data=detected_data,
        source_reference=_string_or_none(
            _first_present(
                data,
                (
                    "source_reference",
                    "image_id",
                    "file_id",
                    "document_id",
                    "id",
                ),
            )
        ),
    )


# ---------------------------------------------------------------------------
# ACOUSTIC ADAPTER
# ---------------------------------------------------------------------------

def adapt_acoustic_result(
    result: Any,
    *,
    user_context: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Adapt an existing Acoustic/deepfake analysis result.

    No Acoustic model is called here.
    """

    raw_data = _to_dict(result)

    if not raw_data:
        return None

    data = _merged_result_data(raw_data)
    data = _normalize_acoustic_fields(data)
    data = _normalize_common_identity_fields(data)
    data = _normalize_relationship_fields(data)

    confidence = _float_or_none(
        _first_present(
            data,
            (
                "confidence",
                "analysis_confidence",
                "detection_confidence",
            ),
        )
    )

    detected_data = deepcopy(data)

    if user_context:
        detected_data.setdefault(
            "user_context",
            user_context,
        )

    return make_evidence(
        source="acoustic",
        evidence_type="acoustic_analysis",
        title="Acoustic analysis",
        description="Existing Acoustic analysis supplied to TrustPath.",
        content=_string_or_none(data.get("extracted_text") or data.get("processed_text") or data.get("text")),
        user_context=user_context,
        status="detected",
        confidence=confidence,
        detected_data=detected_data,
        source_reference=_string_or_none(
            _first_present(
                data,
                (
                    "source_reference",
                    "audio_id",
                    "file_id",
                    "recording_id",
                    "id",
                ),
            )
        ),
    )


# ---------------------------------------------------------------------------
# BIOMETRIC ADAPTER
# ---------------------------------------------------------------------------

def adapt_biometric_result(
    result: Any,
    *,
    user_context: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Adapt the optional Biometric verification result.

    When there is no biometric result, this function returns None.

    No biometric model is called here.
    """

    raw_data = _to_dict(result)

    if not raw_data:
        return None

    data = _merged_result_data(raw_data)

    # A voice-file analysis can contain both Acoustic and Biometric sections.
    # For a TrustPath biometric evidence item, retain only identity-verification
    # signals so Acoustic threat/scenario findings are not duplicated as
    # biometric findings/events.
    biometric_data = {}
    biometric_section = data.get("biometric_forensics")
    if isinstance(biometric_section, dict):
        biometric_data.update(deepcopy(biometric_section))

    for key in (
        "match",
        "is_match",
        "verified",
        "biometric_verified",
        "identity_match",
        "similarity",
        "similarity_score",
        "match_score",
        "verification_score",
        "threshold",
        "similarity_threshold",
        "match_threshold",
        "claimed_identity",
        "impersonation_risk",
        "classification_label",
        "trustpath_reference_target_id",
    ):
        if key in data and data[key] is not None:
            biometric_data.setdefault(key, deepcopy(data[key]))

    biometric_data = _normalize_biometric_fields(biometric_data)
    biometric_data = _normalize_common_identity_fields(biometric_data)
    biometric_data = _normalize_relationship_fields(biometric_data)

    confidence = _float_or_none(
        _first_present(
            biometric_data,
            (
                "confidence",
                "match_confidence",
                "similarity_confidence",
                "verification_confidence",
            ),
        )
    )

    detected_data = deepcopy(biometric_data)

    if user_context:
        detected_data.setdefault(
            "user_context",
            user_context,
        )

    return make_evidence(
        source="biometric",
        evidence_type="biometric_verification",
        title="Biometric verification",
        description="Optional Biometric verification supplied to TrustPath.",
        content=_string_or_none(data.get("extracted_text") or data.get("text")),
        user_context=user_context,
        status="detected",
        confidence=confidence,
        detected_data=detected_data,
        source_reference=_string_or_none(
            _first_present(
                data,
                (
                    "source_reference",
                    "audio_id",
                    "voice_id",
                    "recording_id",
                    "id",
                ),
            )
        ),
    )


# ---------------------------------------------------------------------------
# USER VISUAL ANNOTATION ADAPTER
# ---------------------------------------------------------------------------

def adapt_user_visual_annotation(
    *,
    evidence_id: Optional[str],
    label: Optional[str],
    context: Optional[str],
) -> Optional[Dict[str, Any]]:
    """
    Convert user-provided visual context into an annotation object.

    Example:
        label   = "Person"
        context = "Rahul, my manager"

    IMPORTANT:
        This remains a USER CLAIM / ANNOTATION.

        It must not automatically be represented as a verified fact.
    """

    if not context and not label:
        return None

    return {
        "annotation_id": None,
        "evidence_id": evidence_id,
        "label": label,
        "context": context or "",
        "created_at": None,
        "source": "user",
        "status": "user_provided",
    }


# ---------------------------------------------------------------------------
# EVIDENCE VALIDATION / NORMALIZATION
# ---------------------------------------------------------------------------

def _normalize_existing_evidence_item(
    item: Any,
    *,
    index: int,
) -> Optional[Dict[str, Any]]:
    """
    Accept an already-normalized TrustPath evidence object.

    This is useful when direct TrustPath evidence and adapted module
    evidence are mixed together.
    """

    if not isinstance(item, dict):
        return None

    copied = deepcopy(item)

    source = copied.get("source")

    if source not in SUPPORTED_SOURCES:
        return None

    if not copied.get("evidence_id"):
        copied["evidence_id"] = f"adapter_{source}_{index}"

    if not isinstance(
        copied.get("detected_data"),
        dict,
    ):
        copied["detected_data"] = {
            "value": copied.get("detected_data")
        }

    if "related_entity_ids" not in copied:
        copied["related_entity_ids"] = []

    if "related_event_ids" not in copied:
        copied["related_event_ids"] = []

    return copied


# ---------------------------------------------------------------------------
# COMPLETE ADAPTER ENTRY POINT
# ---------------------------------------------------------------------------

def adapt_existing_results(
    *,
    text: Any = None,
    visual: Any = None,
    acoustic: Any = None,
    biometric: Any = None,
    user_context: Optional[str] = None,
    visual_context: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Normalize all available existing-module results into the TrustPath
    contract.

    Any module may be absent.

    Biometric is optional.

    Returns:

        {
            "evidence": [...],
            "input_sources": [...],
            "user_context": ...
        }
    """

    evidence: List[Dict[str, Any]] = []

    text_evidence = adapt_text_result(
        text,
        user_context=user_context,
    )

    if text_evidence is not None:
        evidence.append(text_evidence)

    visual_evidence = adapt_visual_result(
        visual,
        user_context=user_context,
        visual_context=visual_context,
    )

    if visual_evidence is not None:
        evidence.append(visual_evidence)

    acoustic_evidence = adapt_acoustic_result(
        acoustic,
        user_context=user_context,
    )

    if acoustic_evidence is not None:
        evidence.append(acoustic_evidence)

    # Biometric evidence must only come from the explicit biometric input.
    # Acoustic analysis may contain embedded biometric_forensics metadata.
    # That metadata stays inside Acoustic evidence unless a separate biometric
    # branch was actually supplied by the TrustPath caller.
    biometric_input = biometric
    # Do not infer standalone biometric evidence from acoustic output.
    biometric_evidence = adapt_biometric_result(
        biometric_input,
        user_context=user_context,
    )

    if biometric_evidence is not None:
        evidence.append(biometric_evidence)

    # Assign IDs only after all evidence has been collected.
    for index, item in enumerate(evidence, start=1):
        item["evidence_id"] = _new_evidence_id(
            item["source"],
            index,
        )

    input_sources: List[str] = []

    for item in evidence:
        source = item.get("source")

        if source and source not in input_sources:
            input_sources.append(source)

    return {
        "evidence": evidence,
        "input_sources": input_sources,
        "user_context": user_context,
    }


# ---------------------------------------------------------------------------
# GENERIC DICTIONARY ADAPTER
# ---------------------------------------------------------------------------

def adapt_existing_analysis_dict(
    existing_analysis: Optional[Dict[str, Any]],
    *,
    user_context: Optional[str] = None,
    visual_context: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Adapt the generic existing_analysis dictionary used by TrustPath.

    Expected optional keys:

        {
            "text": {...},
            "visual": {...},
            "acoustic": {...},
            "biometric": {...}
        }

    Unknown top-level keys are deliberately ignored rather than guessed.
    """

    if not isinstance(existing_analysis, dict):
        return {
            "evidence": [],
            "input_sources": [],
            "user_context": user_context,
        }

    return adapt_existing_results(
        text=existing_analysis.get("text"),
        visual=existing_analysis.get("visual"),
        acoustic=existing_analysis.get("acoustic"),
        biometric=existing_analysis.get("biometric"),
        user_context=user_context,
        visual_context=visual_context,
    )


# ---------------------------------------------------------------------------
# DIRECT TRUSTPATH EVIDENCE NORMALIZATION
# ---------------------------------------------------------------------------

def normalize_direct_evidence(
    evidence: Optional[List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """
    Normalize evidence supplied directly to TrustPath.

    Direct evidence is preserved rather than reinterpreted.
    """

    normalized: List[Dict[str, Any]] = []

    for index, item in enumerate(
        evidence or [],
        start=1,
    ):
        normalized_item = _normalize_existing_evidence_item(
            item,
            index=index,
        )

        if normalized_item is not None:
            normalized.append(normalized_item)

    return normalized


# ---------------------------------------------------------------------------
# SAFE MERGE HELPERS
# ---------------------------------------------------------------------------

def merge_trustpath_evidence(
    direct_evidence: Optional[List[Dict[str, Any]]],
    adapted_evidence: Optional[List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """
    Merge explicitly supplied TrustPath evidence with adapted evidence.

    Direct evidence comes first.

    No deduplication is performed because two visually similar pieces of
    evidence may legitimately represent separate artifacts.
    """

    result: List[Dict[str, Any]] = []

    for item in normalize_direct_evidence(
        direct_evidence,
    ):
        result.append(
            deepcopy(item)
        )

    for item in adapted_evidence or []:
        if isinstance(item, dict):
            result.append(
                deepcopy(item)
            )

    return result


# ---------------------------------------------------------------------------
# EXPORTS
# ---------------------------------------------------------------------------

__all__ = [
    "SUPPORTED_SOURCES",
    "make_evidence",
    "adapt_text_result",
    "adapt_visual_result",
    "adapt_acoustic_result",
    "adapt_biometric_result",
    "adapt_user_visual_annotation",
    "adapt_existing_results",
    "adapt_existing_analysis_dict",
    "normalize_direct_evidence",
    "merge_trustpath_evidence",
]