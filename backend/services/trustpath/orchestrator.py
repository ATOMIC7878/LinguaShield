"""
TrustPath Orchestrator
======================

Coordinates the isolated TrustPath reasoning pipeline.

Pipeline
--------
Existing LinguaShield analysis
        ↓
    adapter.py
        ↓
    entities.py
        ↓
 relationships.py
        ↓
 contradictions.py
        ↓
    scenario.py
        ↓
      story.py
        ↓
 assessment + recommendation
        ↓
 unified TrustPath result

Important
---------
This module only orchestrates TrustPath modules.

It does NOT:
- modify Text detection
- modify Visual detection
- modify Acoustic detection
- modify Biometric detection
- modify Live detection
- rerun frozen models
- replace existing model output
- use an LLM
"""

from __future__ import annotations

import inspect
from typing import Any, Dict, List, Optional, Sequence

from .adapter import adapt_existing_analysis_dict
from . import entities as entities_module
from . import relationships as relationships_module
from . import scenario as scenario_module
from . import contradictions as contradictions_module
from .story import build_trustpath_story


# ===========================================================================
# BASIC HELPERS
# ===========================================================================

def _safe_text(value: Any) -> Optional[str]:
    """Return cleaned text or None."""
    if value is None:
        return None

    text = str(value).strip()

    return text if text else None


def _first_non_empty(*values: Any) -> Optional[str]:
    """Return the first non-empty scalar value."""
    for value in values:
        text = _safe_text(value)

        if text:
            return text

    return None


def _unique(values: Sequence[Any]) -> List[Any]:
    """Preserve order while removing duplicates."""
    result: List[Any] = []
    seen = set()

    for value in values:
        marker = repr(value)

        if marker in seen:
            continue

        seen.add(marker)
        result.append(value)

    return result


def _string_list(value: Any) -> List[str]:
    """Normalize scalar/sequence values into a list of strings."""
    if value is None:
        return []

    if isinstance(value, str):
        text = value.strip()

        return [text] if text else []

    if isinstance(value, (list, tuple, set)):
        result: List[str] = []

        for item in value:
            text = _safe_text(item)

            if text:
                result.append(text)

        return _unique(result)

    text = _safe_text(value)

    return [text] if text else []


def _evidence_ids(value: Any) -> List[str]:
    """Normalize evidence IDs."""
    return _string_list(value)


def _as_dict(value: Any) -> Dict[str, Any]:
    """
    Convert a dict-like/Pydantic-like result to a dictionary.
    """
    if isinstance(value, dict):
        return value

    model_dump = getattr(
        value,
        "model_dump",
        None,
    )

    if callable(model_dump):
        try:
            dumped = model_dump()

            if isinstance(
                dumped,
                dict,
            ):
                return dumped
        except Exception:
            pass

    return {}


def _as_list(value: Any) -> List[Any]:
    """Normalize result collections."""
    if value is None:
        return []

    if isinstance(
        value,
        list,
    ):
        return value

    if isinstance(
        value,
        tuple,
    ):
        return list(value)

    return [value]


# ===========================================================================
# ADAPTER
# ===========================================================================

def _normalize_reasoning_input(
    analysis: Optional[Dict[str, Any]],
    user_context: Any = None,
) -> Dict[str, Any]:
    """
    Convert existing analysis into normalized TrustPath evidence.
    """
    if not isinstance(
        analysis,
        dict,
    ):
        analysis = {}

    normalized = adapt_existing_analysis_dict(
        analysis,
        user_context=user_context,
    )

    if not isinstance(
        normalized,
        dict,
    ):
        return {}

    return normalized


def _extract_evidence(
    normalized: Dict[str, Any],
) -> List[Any]:
    evidence = normalized.get(
        "evidence"
    )

    return _as_list(
        evidence
    )


def _extract_user_context(
    normalized: Dict[str, Any],
) -> Any:
    return normalized.get(
        "user_context"
    )


def _extract_events(
    normalized: Dict[str, Any],
) -> List[Any]:
    """
    Extract explicit event information from normalized TrustPath evidence.

    The current adapter preserves detector fields inside each evidence item's
    ``detected_data`` mapping. It does not create a separate top-level
    ``events`` collection. Therefore, the orchestrator derives events here
    only when the evidence explicitly contains event fields.

    This is intentionally conservative: no chronology or event is invented
    from a classification, intent, or free-form sentence alone.
    """

    direct_events = normalized.get(
        "events"
    )

    if direct_events is not None:
        normalized_events = _as_list(
            direct_events
        )

        if normalized_events:
            return normalized_events

    evidence = _as_list(
        normalized.get(
            "evidence",
            [],
        )
    )

    events: List[Dict[str, Any]] = []

    for item_index, item in enumerate(
        evidence,
        start=1,
    ):
        item_dict = _as_dict(
            item
        )

        if not item_dict:
            continue

        detected_data = item_dict.get(
            "detected_data",
            {},
        )

        if not isinstance(
            detected_data,
            dict,
        ):
            continue

        event_title = _first_non_empty(
            detected_data.get("event_title"),
            detected_data.get("event_name"),
        )

        event_description = _first_non_empty(
            detected_data.get("event_description"),
            detected_data.get("event_summary"),
        )

        # Do not manufacture an event when no explicit event information
        # was supplied by the upstream evidence.
        if not event_title and not event_description:
            continue

        event_type = _first_non_empty(
            detected_data.get("event_type"),
            "observed",
        ) or "observed"

        evidence_id = _first_non_empty(
            item_dict.get("evidence_id"),
            item_dict.get("id"),
        )

        event_confidence = detected_data.get(
            "event_confidence"
        )

        try:
            if event_confidence is not None:
                event_confidence = float(
                    event_confidence
                )

                if event_confidence > 1.0:
                    event_confidence = event_confidence / 100.0

                event_confidence = max(
                    0.0,
                    min(
                        1.0,
                        event_confidence,
                    ),
                )
        except (
            TypeError,
            ValueError,
        ):
            event_confidence = None

        events.append(
            {
                "event_id": f"event_{item_index}",
                "event_type": event_type,
                "title": (
                    event_title
                    or _first_non_empty(
                        item_dict.get("title"),
                        "Observed event",
                    )
                    or "Observed event"
                ),
                "description": (
                    event_description
                    or _first_non_empty(
                        item_dict.get("description"),
                        "",
                    )
                    or ""
                ),
                "sequence": len(events) + 1,
                "evidence_ids": (
                    [evidence_id]
                    if evidence_id
                    else []
                ),
                "confidence": event_confidence,
                "source": item_dict.get(
                    "source"
                ),
            }
        )

    return events


# ===========================================================================
# ENTITY LAYER
# ===========================================================================

def _run_entities(
    evidence: Sequence[Any],
) -> Dict[str, Any]:
    """
    Use whichever public entity API exists in the current entities.py.

    Preferred:
        build_entity_registry()

    Compatibility:
        extract_entities()
    """

    builder = getattr(
        entities_module,
        "build_entity_registry",
        None,
    )

    if callable(
        builder
    ):
        result = builder(
            evidence=evidence,
        )

        result_dict = _as_dict(
            result
        )

        if result_dict:
            return result_dict

    legacy_builder = getattr(
        entities_module,
        "extract_entities",
        None,
    )

    if callable(
        legacy_builder
    ):
        result = legacy_builder(
            evidence,
        )

        if isinstance(
            result,
            list,
        ):
            return {
                "entities": result,
            }

        return _as_dict(
            result
        )

    raise ImportError(
        "entities.py does not expose a supported entity builder."
    )


def _extract_entities(
    entity_result: Dict[str, Any],
) -> List[Any]:
    return _as_list(
        entity_result.get(
            "entities",
            [],
        )
    )


# ===========================================================================
# RELATIONSHIP LAYER
# ===========================================================================

def _run_relationships(
    evidence: Sequence[Any],
    entities: Sequence[Any],
) -> Dict[str, Any]:
    """
    Use the relationship API that actually exists in relationships.py.
    """

    builder = getattr(
        relationships_module,
        "build_relationships",
        None,
    )

    if callable(
        builder
    ):
        result = builder(
            entities=entities,
            evidence=evidence,
        )

        if isinstance(
            result,
            list,
        ):
            return {
                "relationships": result,
            }

        result_dict = _as_dict(
            result
        )

        if result_dict:
            return result_dict

    compatibility_builder = getattr(
        relationships_module,
        "build_relationship_graph",
        None,
    )

    if callable(
        compatibility_builder
    ):
        result = compatibility_builder(
            evidence=evidence,
            entities=entities,
        )

        if isinstance(
            result,
            list,
        ):
            return {
                "relationships": result,
            }

        result_dict = _as_dict(
            result
        )

        if result_dict:
            return result_dict

    raise ImportError(
        "relationships.py does not expose a supported relationship builder."
    )


def _extract_relationships(
    relationship_result: Dict[str, Any],
) -> List[Any]:
    return _as_list(
        relationship_result.get(
            "relationships",
            [],
        )
    )


# ===========================================================================
# GENERIC SIGNATURE-AWARE FUNCTION CALL
# ===========================================================================

def _call_compatible(
    function: Any,
    available_values: Dict[str, Any],
) -> Any:
    """
    Call a function using only parameters accepted by its actual signature.

    This is important because TrustPath modules may be at slightly different
    isolated revisions while retaining the same overall role.

    Example:
        If reconstruct_scenario() accepts:
            evidence, entities, relationships

        only those values are passed.

        If it accepts:
            evidence, events, contradictions

        those are passed instead.

    No module source needs to be changed.
    """

    try:
        signature = inspect.signature(
            function
        )
    except (
        TypeError,
        ValueError,
    ):
        # Fallback for unusual callables.
        return function(
            **available_values
        )

    parameters = signature.parameters

    kwargs: Dict[str, Any] = {}

    for parameter_name, parameter in parameters.items():

        # Skip *args and **kwargs.
        if parameter.kind in {
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }:
            continue

        if parameter_name in available_values:
            kwargs[
                parameter_name
            ] = available_values[
                parameter_name
            ]

    # Check required parameters.
    missing_required: List[str] = []

    for parameter_name, parameter in parameters.items():

        if parameter.kind in {
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }:
            continue

        if (
            parameter.default
            is inspect.Parameter.empty
            and parameter_name
            not in kwargs
        ):
            missing_required.append(
                parameter_name
            )

    if missing_required:
        raise TypeError(
            "Function requires unsupported/missing parameters: "
            + ", ".join(
                missing_required
            )
        )

    return function(
        **kwargs
    )


# ===========================================================================
# SCENARIO LAYER
# ===========================================================================

def _run_scenario(
    evidence: Sequence[Any],
    entities: Sequence[Any],
    relationships: Sequence[Any],
    events: Sequence[Any],
    contradictions: Sequence[Any],
) -> Dict[str, Any]:
    """
    Run reconstruct_scenario() using the exact signature of the currently
    installed scenario.py.

    This avoids hard-coding an incorrect function interface.
    """

    builder = getattr(
        scenario_module,
        "reconstruct_scenario",
        None,
    )

    if not callable(
        builder
    ):
        raise ImportError(
            "scenario.py does not expose 'reconstruct_scenario'."
        )

    available_values: Dict[str, Any] = {
        "evidence": evidence,
        "entities": entities,
        "relationships": relationships,
        "events": events,
        "contradictions": contradictions,
    }

    result = _call_compatible(
        builder,
        available_values,
    )

    # Pydantic/dataclass/dict result.
    result_dict = _as_dict(
        result
    )

    if result_dict:
        return result_dict

    # Some versions may directly return a list or another object.
    return {
        "scenario": result,
    }


# ===========================================================================
# CONTRADICTION LAYER
# ===========================================================================

def _run_contradictions(
    evidence: Sequence[Any],
    entities: Sequence[Any],
    relationships: Sequence[Any],
    events: Sequence[Any],
) -> Dict[str, Any]:
    """
    Run contradiction detection using the signature supported by the current
    contradictions.py.
    """

    detector = getattr(
        contradictions_module,
        "detect_contradictions",
        None,
    )

    if not callable(
        detector
    ):
        raise ImportError(
            "contradictions.py does not expose 'detect_contradictions'."
        )

    available_values: Dict[str, Any] = {
        "evidence": evidence,
        "entities": entities,
        "relationships": relationships,
        "events": events,
    }

    result = _call_compatible(
        detector,
        available_values,
    )

    if isinstance(
        result,
        list,
    ):
        return {
            "contradictions": result,
        }

    result_dict = _as_dict(
        result
    )

    if result_dict:
        return result_dict

    return {
        "contradictions": [],
    }


def _extract_contradictions(
    contradiction_result: Dict[str, Any],
) -> List[Any]:
    return _as_list(
        contradiction_result.get(
            "contradictions",
            [],
        )
    )


# ===========================================================================
# SCENARIO EXTRACTION
# ===========================================================================

def _extract_scenario(
    scenario_result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Normalize nested/flat scenario results.
    """
    if not isinstance(
        scenario_result,
        dict,
    ):
        return {}

    nested = scenario_result.get(
        "scenario"
    )

    if isinstance(
        nested,
        dict,
    ):
        return dict(
            nested
        )

    return dict(
        scenario_result
    )


def _scenario_type(
    scenario_result: Dict[str, Any],
) -> str:
    scenario = _extract_scenario(
        scenario_result
    )

    return (
        _first_non_empty(
            scenario.get(
                "scenario_type"
            ),
            "unknown",
        )
        or "unknown"
    )


def _scenario_confidence(
    scenario_result: Dict[str, Any],
) -> float:
    scenario = _extract_scenario(
        scenario_result
    )

    value = scenario.get(
        "confidence",
        0.0,
    )

    try:
        converted = float(
            value
        )
    except (
        TypeError,
        ValueError,
    ):
        return 0.0

    return max(
        0.0,
        min(
            1.0,
            converted,
        ),
    )


# ===========================================================================
# RISK
# ===========================================================================

def _derive_risk_level(
    scenario_result: Dict[str, Any],
    contradiction_result: Dict[str, Any],
) -> str:
    """
    Derive display-level risk without modifying scenario.py.
    """

    scenario = _extract_scenario(
        scenario_result
    )

    explicit_risk = _first_non_empty(
        scenario_result.get(
            "risk_level"
        ),
        scenario_result.get(
            "risk"
        ),
        scenario.get(
            "risk_level"
        ),
    )

    if explicit_risk:
        return explicit_risk.upper()

    contradiction_summary = (
        contradiction_result.get(
            "summary",
            {},
        )
    )

    if isinstance(
        contradiction_summary,
        dict,
    ):
        highest = _safe_text(
            contradiction_summary.get(
                "highest_severity"
            )
        )

        if highest:
            highest = highest.upper()

            if highest in {
                "CRITICAL",
                "HIGH",
            }:
                return highest

    scenario_type = _scenario_type(
        scenario_result
    ).lower()

    if scenario_type in {
        "possible_impersonation",
        "possible_payment_diversion",
        "possible_credential_request",
        "possible_account_takeover",
        "possible_fraud",
    }:
        return "HIGH"

    if scenario_type in {
        "possible_social_engineering",
        "possible_document_manipulation",
        "evidence_inconsistency",
    }:
        return "MEDIUM"

    if scenario_type == "communication_event":
        return "LOW"

    return "UNKNOWN"


# ===========================================================================
# TRUST SCORE
# ===========================================================================

def _derive_trust_score(
    scenario_result: Dict[str, Any],
    contradiction_result: Dict[str, Any],
) -> float:
    """
    Transparent presentation score.

    This is not an ML model.
    """

    scenario_confidence = _scenario_confidence(
        scenario_result
    )

    score = 50.0

    score -= (
        scenario_confidence * 25.0
    )

    summary = contradiction_result.get(
        "summary",
        {},
    )

    if isinstance(
        summary,
        dict,
    ):
        by_severity = summary.get(
            "by_severity",
            {},
        )

        if isinstance(
            by_severity,
            dict,
        ):
            try:
                low = int(
                    by_severity.get(
                        "LOW",
                        0,
                    )
                )
            except (
                TypeError,
                ValueError,
            ):
                low = 0

            try:
                medium = int(
                    by_severity.get(
                        "MEDIUM",
                        0,
                    )
                )
            except (
                TypeError,
                ValueError,
            ):
                medium = 0

            try:
                high = int(
                    by_severity.get(
                        "HIGH",
                        0,
                    )
                )
            except (
                TypeError,
                ValueError,
            ):
                high = 0

            try:
                critical = int(
                    by_severity.get(
                        "CRITICAL",
                        0,
                    )
                )
            except (
                TypeError,
                ValueError,
            ):
                critical = 0

            score -= (
                low * 3.0
            )

            score -= (
                medium * 7.0
            )

            score -= (
                high * 12.0
            )

            score -= (
                critical * 20.0
            )

    return round(
        max(
            0.0,
            min(
                100.0,
                score,
            ),
        ),
        2,
    )


# ===========================================================================
# ASSESSMENT
# ===========================================================================

def _build_assessment(
    scenario_result: Dict[str, Any],
    contradiction_result: Dict[str, Any],
) -> Dict[str, Any]:
    scenario_confidence = _scenario_confidence(
        scenario_result
    )

    summary = contradiction_result.get(
        "summary",
        {},
    )

    contradiction_count = 0

    if isinstance(
        summary,
        dict,
    ):
        try:
            contradiction_count = int(
                summary.get(
                    "total",
                    0,
                )
            )
        except (
            TypeError,
            ValueError,
        ):
            contradiction_count = 0

    scenario_type = _scenario_type(
        scenario_result
    ).lower()

    if scenario_type == "possible_impersonation":
        identity_confidence = 0.32
        relationship_confidence = 0.38

    elif scenario_type == "possible_social_engineering":
        identity_confidence = 0.50
        relationship_confidence = 0.45

    else:
        identity_confidence = 0.50
        relationship_confidence = 0.50

    return {
        "identity_confidence": round(
            identity_confidence,
            4,
        ),
        "relationship_confidence": round(
            relationship_confidence,
            4,
        ),
        "scenario_confidence": round(
            scenario_confidence,
            4,
        ),
        "contradiction_count": contradiction_count,
        "trust_score": _derive_trust_score(
            scenario_result,
            contradiction_result,
        ),
        "risk_level": _derive_risk_level(
            scenario_result,
            contradiction_result,
        ),
        "method": (
            "transparent_trustpath_aggregation"
        ),
    }


# ===========================================================================
# RECOMMENDATION
# ===========================================================================

def _build_recommendation(
    assessment: Dict[str, Any],
    scenario_result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build a verification-oriented recommendation.
    """

    scenario_type = _scenario_type(
        scenario_result
    ).lower()

    risk_level = (
        _safe_text(
            assessment.get(
                "risk_level"
            )
        )
        or "UNKNOWN"
    ).upper()

    if (
        "payment" in scenario_type
        or "fraud" in scenario_type
        or "impersonation" in scenario_type
    ):
        action = (
            "Pause the transaction and independently verify "
            "the sender and payment details."
        )

    elif (
        "credential" in scenario_type
        or "account_takeover" in scenario_type
    ):
        action = (
            "Do not share credentials, OTPs, PINs, or verification "
            "codes. Verify the request independently."
        )

    elif (
        "document_manipulation"
        in scenario_type
    ):
        action = (
            "Do not approve the document until its source and "
            "critical details have been independently verified."
        )

    else:
        action = (
            "Pause before acting and independently verify the "
            "identity, relationship, and request."
        )

    try:
        contradiction_count = int(
            assessment.get(
                "contradiction_count",
                0,
            )
            or 0
        )
    except (
        TypeError,
        ValueError,
    ):
        contradiction_count = 0

    return {
        "action": action,
        "urgency": (
            "high"
            if risk_level in {
                "HIGH",
                "CRITICAL",
            }
            else "normal"
        ),
        "risk_level": risk_level,
        "contradictions_present": (
            contradiction_count > 0
        ),
        "independent_verification_required": True,
        "reason": (
            "Recommendation is derived from the reconstructed "
            "scenario and detected evidence conflicts."
        ),
    }


# ===========================================================================
# TRACEABILITY
# ===========================================================================

def _record_id(
    record: Any,
    *names: str,
) -> Optional[str]:
    """
    Read an ID from dict/Pydantic-like records.
    """
    for name in names:

        value = _field_value(
            record,
            name,
        )

        text = _safe_text(
            value
        )

        if text:
            return text

    return None


def _field_value(
    record: Any,
    name: str,
) -> Any:
    if isinstance(
        record,
        dict,
    ):
        return record.get(
            name
        )

    try:
        value = getattr(
            record,
            name,
            None,
        )
    except Exception:
        value = None

    if value is not None:
        return value

    model_dump = getattr(
        record,
        "model_dump",
        None,
    )

    if callable(
        model_dump
    ):
        try:
            dumped = model_dump()

            if isinstance(
                dumped,
                dict,
            ):
                return dumped.get(
                    name
                )
        except Exception:
            pass

    return None


def _build_traceability(
    evidence: Sequence[Any],
    entities: Sequence[Any],
    relationships: Sequence[Any],
    contradictions: Sequence[Any],
    story: Dict[str, Any],
) -> Dict[str, Any]:
    evidence_ids = [
        evidence_id
        for evidence_id in (
            _record_id(
                item,
                "evidence_id",
                "id",
            )
            for item in evidence
        )
        if evidence_id
    ]

    entity_ids = [
        entity_id
        for entity_id in (
            _record_id(
                item,
                "entity_id",
                "id",
                "object_id",
            )
            for item in entities
        )
        if entity_id
    ]

    relationship_ids = [
        relationship_id
        for relationship_id in (
            _record_id(
                item,
                "relationship_id",
                "id",
            )
            for item in relationships
        )
        if relationship_id
    ]

    contradiction_ids = [
        contradiction_id
        for contradiction_id in (
            _record_id(
                item,
                "contradiction_id",
                "id",
            )
            for item in contradictions
        )
        if contradiction_id
    ]

    story_traceability = {}

    if isinstance(
        story,
        dict,
    ):
        raw = story.get(
            "traceability",
            {},
        )

        if isinstance(
            raw,
            dict,
        ):
            story_traceability = raw

    return {
        "evidence_count": len(
            evidence_ids
        ),
        "entity_count": len(
            entity_ids
        ),
        "relationship_count": len(
            relationship_ids
        ),
        "contradiction_count": len(
            contradiction_ids
        ),
        "evidence_ids": _unique(
            evidence_ids
        ),
        "entity_ids": _unique(
            entity_ids
        ),
        "relationship_ids": _unique(
            relationship_ids
        ),
        "contradiction_ids": _unique(
            contradiction_ids
        ),
        "story_traceability": story_traceability,
    }


# ===========================================================================
# MAIN ORCHESTRATOR
# ===========================================================================

def run_trustpath(
    analysis: Optional[
        Dict[str, Any]
    ] = None,
    *,
    user_context: Any = None,
) -> Dict[str, Any]:
    """
    Execute the complete TrustPath pipeline.
    """

    # -----------------------------------------------------------------------
    # 1. Normalize existing analysis
    # -----------------------------------------------------------------------

    normalized = _normalize_reasoning_input(
        analysis,
        user_context=user_context,
    )

    evidence = _extract_evidence(
        normalized
    )

    effective_user_context = (
        _extract_user_context(
            normalized
        )
    )

    # -----------------------------------------------------------------------
    # 2. Entities
    # -----------------------------------------------------------------------

    entity_result = _run_entities(
        evidence
    )

    entities = _extract_entities(
        entity_result
    )

    # -----------------------------------------------------------------------
    # 3. Relationships
    # -----------------------------------------------------------------------

    relationship_result = _run_relationships(
        evidence,
        entities,
    )

    relationships = _extract_relationships(
        relationship_result
    )

    # -----------------------------------------------------------------------
    # 4. Events
    # -----------------------------------------------------------------------

    events = _extract_events(
        normalized
    )

    # -----------------------------------------------------------------------
    # 5. Contradictions
    #
    # Run independently because the current contradictions.py already
    # supports the normalized graph.
    # -----------------------------------------------------------------------

    contradiction_result = _run_contradictions(
        evidence,
        entities,
        relationships,
        events,
    )

    contradictions = _extract_contradictions(
        contradiction_result
    )

    # -----------------------------------------------------------------------
    # 6. Scenario
    #
    # Signature is detected dynamically from scenario.py.
    # -----------------------------------------------------------------------

    scenario_result = _run_scenario(
        evidence,
        entities,
        relationships,
        events,
        contradictions,
    )

    # -----------------------------------------------------------------------
    # 7. Assessment
    # -----------------------------------------------------------------------

    assessment = _build_assessment(
        scenario_result,
        contradiction_result,
    )

    # -----------------------------------------------------------------------
    # 8. Recommendation
    # -----------------------------------------------------------------------

    recommendation = _build_recommendation(
        assessment,
        scenario_result,
    )

    # -----------------------------------------------------------------------
    # 9. Story
    # -----------------------------------------------------------------------

    story = build_trustpath_story(
        evidence=evidence,
        entities=entities,
        relationships=relationships,
        events=events,
        contradictions=contradictions,
        scenario=scenario_result,
        risk_level=assessment.get(
            "risk_level"
        ),
        user_context=effective_user_context,
    )

    # -----------------------------------------------------------------------
    # 10. Traceability
    # -----------------------------------------------------------------------

    traceability = _build_traceability(
        evidence,
        entities,
        relationships,
        contradictions,
        story,
    )

    # -----------------------------------------------------------------------
    # 11. Unified result
    # -----------------------------------------------------------------------

    return {
        "normalized": normalized,

        "evidence": evidence,

        "entities": entities,
        "entity_result": entity_result,

        "relationships": relationships,
        "relationship_result": relationship_result,

        "events": events,

        "scenario": scenario_result,

        "contradictions": contradictions,
        "contradiction_result": contradiction_result,

        "assessment": assessment,

        "recommendation": recommendation,

        "story": story,

        "traceability": traceability,

        "metadata": {
            "orchestrator": (
                "trustpath.orchestrator"
            ),
            "deterministic": True,
            "llm_used": False,
            "frozen_modules_modified": False,

            "pipeline": [
                "adapter",
                "entities",
                "relationships",
                "events",
                "contradictions",
                "scenario",
                "assessment",
                "recommendation",
                "story",
            ],

            "signature_aware_dispatch": True,
        },
    }


# ===========================================================================
# ALIAS
# ===========================================================================

def orchestrate_trustpath(
    analysis: Optional[
        Dict[str, Any]
    ] = None,
    *,
    user_context: Any = None,
) -> Dict[str, Any]:
    """
    Alias for run_trustpath().
    """
    return run_trustpath(
        analysis,
        user_context=user_context,
    )


# ===========================================================================
# VALIDATION
# ===========================================================================

def validate_orchestration_result(
    result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Validate the high-level TrustPath result.
    """

    if not isinstance(
        result,
        dict,
    ):
        return {
            "valid": False,
            "errors": [
                "Orchestration result must be a dictionary."
            ],
        }

    errors: List[str] = []

    required_fields = (
        "normalized",
        "evidence",
        "entities",
        "relationships",
        "events",
        "scenario",
        "contradictions",
        "assessment",
        "recommendation",
        "story",
        "traceability",
        "metadata",
    )

    # -----------------------------------------------------------------------
    # Required fields
    # -----------------------------------------------------------------------

    for field in required_fields:
        if field not in result:
            errors.append(
                f"Missing required field: {field}"
            )

    # -----------------------------------------------------------------------
    # Lists
    # -----------------------------------------------------------------------

    for field in (
        "evidence",
        "entities",
        "relationships",
        "events",
        "contradictions",
    ):
        if field in result:
            if not isinstance(
                result[field],
                list,
            ):
                errors.append(
                    f"{field} must be a list."
                )

    # -----------------------------------------------------------------------
    # Dictionaries
    # -----------------------------------------------------------------------

    for field in (
        "normalized",
        "scenario",
        "assessment",
        "recommendation",
        "story",
        "traceability",
        "metadata",
    ):
        if field in result:
            if not isinstance(
                result[field],
                dict,
            ):
                errors.append(
                    f"{field} must be a dictionary."
                )

    # -----------------------------------------------------------------------
    # Evidence IDs
    # -----------------------------------------------------------------------

    known_evidence_ids = {
        evidence_id
        for evidence_id in (
            _record_id(
                item,
                "evidence_id",
                "id",
            )
            for item in result.get(
                "evidence",
                [],
            )
        )
        if evidence_id
    }

    traceability = result.get(
        "traceability",
        {},
    )

    if isinstance(
        traceability,
        dict,
    ):
        reported_ids = set(
            _string_list(
                traceability.get(
                    "evidence_ids"
                )
            )
        )

        unknown_ids = (
            reported_ids
            - known_evidence_ids
        )

        if unknown_ids:
            errors.append(
                "Traceability contains unknown evidence IDs: "
                + ", ".join(
                    sorted(
                        unknown_ids
                    )
                )
            )

    # -----------------------------------------------------------------------
    # Story traceability
    # -----------------------------------------------------------------------

    story = result.get(
        "story",
        {},
    )

    if isinstance(
        story,
        dict,
    ):
        story_traceability = story.get(
            "traceability",
            {},
        )

        if isinstance(
            story_traceability,
            dict,
        ):
            if (
                story_traceability.get(
                    "valid"
                )
                is False
            ):
                errors.append(
                    "Story evidence traceability is invalid."
                )

    # -----------------------------------------------------------------------
    # Contradiction traceability
    # -----------------------------------------------------------------------

    contradiction_result = result.get(
        "contradiction_result",
        {},
    )

    if isinstance(
        contradiction_result,
        dict,
    ):
        contradiction_traceability = (
            contradiction_result.get(
                "traceability",
                {},
            )
        )

        if isinstance(
            contradiction_traceability,
            dict,
        ):
            if (
                contradiction_traceability.get(
                    "valid"
                )
                is False
            ):
                errors.append(
                    "Contradiction evidence traceability is invalid."
                )

    # -----------------------------------------------------------------------
    # Metadata
    # -----------------------------------------------------------------------

    metadata = result.get(
        "metadata",
        {},
    )

    if isinstance(
        metadata,
        dict,
    ):
        if metadata.get(
            "frozen_modules_modified"
        ) is True:
            errors.append(
                "TrustPath reports that a frozen module was modified."
            )

    return {
        "valid": len(errors) == 0,
        "errors": errors,
    }


# ===========================================================================
# PUBLIC EXPORTS
# ===========================================================================

__all__ = [
    "run_trustpath",
    "orchestrate_trustpath",
    "validate_orchestration_result",
]