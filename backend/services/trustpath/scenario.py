"""
LinguaShield - TrustPath Scenario Reconstruction Engine
--------------------------------------------------------

Purpose:
    Reconstruct the most defensible communication scenario from the
    structured TrustPath evidence, entities, relationships, and events.

Architecture:

    adapter.py
        ↓
    entities.py
        ↓
    relationships.py
        ↓
    scenario.py
        ↓
    story.py

Important:
    - This module does NOT call Text, Visual, Acoustic, Biometric, or Live
      detection models.
    - It does NOT modify any existing LinguaShield detection module.
    - It does NOT treat user claims as verified facts.
    - It does NOT automatically declare an event to be fraud/scam.
    - It keeps every scenario conclusion traceable to evidence.
    - It is designed so a richer validated reasoning engine can replace or
      extend this logic later without changing the output contract.

Expected location:
    LinguaShield/backend/services/trustpath/scenario.py
"""

from __future__ import annotations

import re
from collections import defaultdict
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
from uuid import uuid4


# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

SCENARIO_TYPES = {
    "unknown",
    "communication_event",
    "evidence_inconsistency",
    "possible_impersonation",
    "possible_payment_diversion",
    "possible_credential_request",
    "possible_social_engineering",
    "possible_document_manipulation",
    "possible_account_takeover",
    "possible_fraud",
}

SCENARIO_RISK_LEVELS = {
    "LOW",
    "MEDIUM",
    "HIGH",
    "CRITICAL",
    "UNKNOWN",
}

RELATIONSHIP_RISK_SIGNALS = {
    "contradicted",
    "unknown",
}

HIGH_RISK_RELATIONS = {
    "requests",
    "requested",
    "transferred_to",
    "paid_to",
    "claims_identity",
    "claims_role",
}

FINANCIAL_TERMS = {
    "payment",
    "transfer",
    "money",
    "invoice",
    "bank",
    "account",
    "transaction",
    "wire",
    "upi",
    "refund",
    "amount",
}

IDENTITY_TERMS = {
    "manager",
    "ceo",
    "director",
    "bank",
    "support",
    "officer",
    "admin",
    "administrator",
    "employee",
    "customer",
    "relative",
    "friend",
    "police",
    "government",
    "agent",
}

CREDENTIAL_TERMS = {
    "otp",
    "password",
    "passcode",
    "pin",
    "login",
    "credential",
    "verification code",
    "security code",
    "authentication",
}

PRESSURE_TERMS = {
    "urgent",
    "immediately",
    "now",
    "asap",
    "deadline",
    "final warning",
    "don't delay",
    "do not delay",
    "emergency",
    "quickly",
}


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def _new_id(prefix: str = "scenario") -> str:
    """Create a readable TrustPath scenario identifier."""
    return f"{prefix}_{uuid4().hex[:12]}"


def _clean_string(value: Any) -> Optional[str]:
    """Return a stripped non-empty string or None."""

    if value is None:
        return None

    if not isinstance(value, str):
        value = str(value)

    value = value.strip()

    return value or None


def _as_dict(value: Any) -> Dict[str, Any]:
    """
    Convert a dictionary/Pydantic/plain object into a dictionary.
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

    legacy_dict = getattr(value, "dict", None)

    if callable(legacy_dict):
        try:
            result = legacy_dict()

            if isinstance(result, dict):
                return deepcopy(result)

        except Exception:
            pass

    raw_dict = getattr(value, "__dict__", None)

    if isinstance(raw_dict, dict):
        return deepcopy(raw_dict)

    return {}


def _safe_float(value: Any) -> Optional[float]:
    """Convert 0..1 or 0..100 confidence-like values into 0..1."""

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


def _clamp(
    value: float,
    minimum: float = 0.0,
    maximum: float = 1.0,
) -> float:
    """Clamp a value to a numeric range."""

    return max(
        minimum,
        min(maximum, value),
    )


def _append_unique(
    target: List[str],
    values: Iterable[str],
) -> None:
    """Append strings while preserving order and uniqueness."""

    for value in values:
        if not value:
            continue

        if value not in target:
            target.append(value)


def _normalize_text(value: Any) -> str:
    """Normalize arbitrary text for lightweight keyword matching."""

    value = _clean_string(value)

    if not value:
        return ""

    value = value.casefold()

    value = re.sub(
        r"\s+",
        " ",
        value,
    )

    return value


def _text_contains_any(
    text: str,
    terms: Set[str],
) -> List[str]:
    """
    Return matching terms from a controlled vocabulary.

    This is not semantic NLP. It is a transparent signal extractor.
    """

    normalized = _normalize_text(text)

    if not normalized:
        return []

    matches: List[str] = []

    for term in terms:
        term_normalized = _normalize_text(term)

        if not term_normalized:
            continue

        if term_normalized in normalized:
            matches.append(term)

    return sorted(
        set(matches)
    )


# ---------------------------------------------------------------------------
# EVIDENCE NORMALIZATION
# ---------------------------------------------------------------------------

def _evidence_text(item: Dict[str, Any]) -> str:
    """
    Build a searchable evidence text from fields that may contain
    communication/scenario clues.
    """

    parts: List[str] = []

    for key in (
        "title",
        "description",
        "user_context",
    ):
        value = _clean_string(
            item.get(key)
        )

        if value:
            parts.append(value)

    detected_data = item.get(
        "detected_data",
        {},
    )

    if isinstance(
        detected_data,
        dict,
    ):
        for key in (
            "message",
            "text",
            "ocr",
            "ocr_text",
            "extracted_text",
            "intent",
            "request",
            "request_type",
            "scenario_summary",
            "current_manipulation",
            "victim_decision",
            "likely_next_action",
            "classification",
            "threat_category",
        ):
            value = detected_data.get(key)

            if isinstance(
                value,
                str,
            ):
                parts.append(value)

    return " ".join(parts)


def _collect_evidence_ids(
    items: Iterable[Dict[str, Any]],
) -> List[str]:
    """Collect evidence IDs from a group of evidence records."""

    result: List[str] = []

    for item in items:
        evidence_id = _clean_string(
            item.get("evidence_id")
        )

        if evidence_id:
            _append_unique(
                result,
                [evidence_id],
            )

    return result


# ---------------------------------------------------------------------------
# EXPLICIT SCENARIO SIGNALS
# ---------------------------------------------------------------------------

def _extract_explicit_scenario(
    evidence: Iterable[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """
    Prefer explicitly supplied scenario information.

    The information may come from:
        - a future detector
        - an adapter
        - a validated downstream model

    It is still traceable to evidence.
    """

    candidates: List[
        Tuple[float, Dict[str, Any]]
    ] = []

    for item in evidence:
        detected_data = item.get(
            "detected_data",
            {},
        )

        if not isinstance(
            detected_data,
            dict,
        ):
            continue

        scenario_type = _clean_string(
            detected_data.get(
                "scenario_type"
            )
        )

        if not scenario_type:
            continue

        confidence = (
            _safe_float(
                detected_data.get(
                    "scenario_confidence"
                )
            )
            or 0.0
        )

        candidates.append(
            (
                confidence,
                item,
            )
        )

    if not candidates:
        return None

    candidates.sort(
        key=lambda pair: pair[0],
        reverse=True,
    )

    confidence, item = candidates[0]

    detected_data = item.get(
        "detected_data",
        {},
    )

    scenario_type = _clean_string(
        detected_data.get(
            "scenario_type"
        )
    ) or "unknown"

    summary = (
        _clean_string(
            detected_data.get(
                "scenario_summary"
            )
        )
        or _clean_string(
            item.get("description")
        )
        or "Scenario supplied by an upstream analysis."
    )

    title = scenario_type.replace(
        "_",
        " ",
    ).title()

    evidence_id = _clean_string(
        item.get("evidence_id")
    )

    return {
        "scenario_type": (
            scenario_type
            if scenario_type in SCENARIO_TYPES
            else "unknown"
        ),
        "title": title,
        "summary": summary,
        "confidence": _clamp(
            confidence
        ),
        "evidence_ids": (
            [evidence_id]
            if evidence_id
            else []
        ),
        "source": "explicit",
    }


# ---------------------------------------------------------------------------
# SIGNAL EXTRACTION
# ---------------------------------------------------------------------------

def extract_scenario_signals(
    evidence: Iterable[Dict[str, Any]],
    relationships: Iterable[Dict[str, Any]],
    events: Iterable[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Extract transparent scenario signals.

    This does not decide the scenario by itself.
    """

    evidence_list = [
        _as_dict(item)
        for item in evidence
    ]

    relationship_list = [
        _as_dict(item)
        for item in relationships
    ]

    event_list = [
        _as_dict(item)
        for item in events
    ]

    combined_text = " ".join(
        _evidence_text(item)
        for item in evidence_list
    )

    financial_matches = _text_contains_any(
        combined_text,
        FINANCIAL_TERMS,
    )

    identity_matches = _text_contains_any(
        combined_text,
        IDENTITY_TERMS,
    )

    credential_matches = _text_contains_any(
        combined_text,
        CREDENTIAL_TERMS,
    )

    pressure_matches = _text_contains_any(
        combined_text,
        PRESSURE_TERMS,
    )

    scenario_types: List[str] = []

    for item in evidence_list:
        detected_data = item.get(
            "detected_data",
            {},
        )

        if not isinstance(
            detected_data,
            dict,
        ):
            continue

        classification = _clean_string(
            detected_data.get(
                "classification"
            )
        )

        if classification:
            normalized_classification = (
                classification.casefold()
            )

            if "credential" in normalized_classification:
                scenario_types.append(
                    "possible_credential_request"
                )

            if (
                "imperson" in
                normalized_classification
            ):
                scenario_types.append(
                    "possible_impersonation"
                )

            if "payment" in normalized_classification:
                scenario_types.append(
                    "possible_payment_diversion"
                )

    relationship_signals: List[str] = []

    for relationship in relationship_list:
        status = _clean_string(
            relationship.get(
                "status"
            )
        )

        relation = _clean_string(
            relationship.get(
                "relation"
            )
        )

        if status:
            relationship_signals.append(
                status
            )

        if relation:
            relationship_signals.append(
                relation
            )

    contradictory_relationship_count = sum(
        1
        for relationship in relationship_list
        if relationship.get("status")
        == "contradicted"
    )

    claimed_relationship_count = sum(
        1
        for relationship in relationship_list
        if relationship.get("status")
        == "claimed"
    )

    high_risk_relationship_count = sum(
        1
        for relationship in relationship_list
        if relationship.get("relation")
        in HIGH_RISK_RELATIONS
    )

    synthetic_voice_count = 0

    biometric_mismatch_count = 0

    for item in evidence_list:
        data = item.get(
            "detected_data",
            {},
        )

        if not isinstance(
            data,
            dict,
        ):
            continue

        synthetic_detected = data.get(
            "synthetic_detected"
        )

        if synthetic_detected is True:
            synthetic_voice_count += 1

        match_value = data.get(
            "match",
            data.get(
                "is_match"
            ),
        )

        if match_value is False:
            biometric_mismatch_count += 1

    return {
        "financial_terms": financial_matches,
        "identity_terms": identity_matches,
        "credential_terms": credential_matches,
        "pressure_terms": pressure_matches,
        "scenario_types": sorted(
            set(
                scenario_types
            )
        ),
        "relationship_signals": sorted(
            set(
                relationship_signals
            )
        ),
        "contradicted_relationships": (
            contradictory_relationship_count
        ),
        "claimed_relationships": (
            claimed_relationship_count
        ),
        "high_risk_relationships": (
            high_risk_relationship_count
        ),
        "synthetic_voice_evidence": (
            synthetic_voice_count
        ),
        "biometric_mismatch_evidence": (
            biometric_mismatch_count
        ),
        "event_count": len(
            event_list
        ),
    }


# ---------------------------------------------------------------------------
# SCENARIO CANDIDATE SCORING
# ---------------------------------------------------------------------------

def _add_candidate_score(
    scores: Dict[str, float],
    evidence_map: Dict[str, List[str]],
    *,
    scenario_type: str,
    score: float,
    evidence_ids: Iterable[str],
) -> None:
    """Add weighted candidate evidence."""

    if score <= 0:
        return

    scores[scenario_type] = (
        scores.get(
            scenario_type,
            0.0,
        )
        + score
    )

    _append_unique(
        evidence_map.setdefault(
            scenario_type,
            [],
        ),
        list(evidence_ids),
    )


def score_scenario_candidates(
    *,
    evidence: Iterable[Dict[str, Any]],
    relationships: Iterable[Dict[str, Any]],
    events: Iterable[Dict[str, Any]],
    signals: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Score scenario candidates using transparent, explainable signals.

    These scores are not machine-learning probabilities.
    They are internal evidence weights used only to rank candidate scenarios.
    """

    evidence_list = [
        _as_dict(item)
        for item in evidence
    ]

    relationship_list = [
        _as_dict(item)
        for item in relationships
    ]

    event_list = [
        _as_dict(item)
        for item in events
    ]

    scores: Dict[str, float] = {}

    evidence_map: Dict[
        str,
        List[str],
    ] = {}

    # -----------------------------------------------------------------------
    # Candidate: possible impersonation
    # -----------------------------------------------------------------------

    identity_terms = set(
        signals.get(
            "identity_terms",
            [],
        )
    )

    if identity_terms:
        identity_evidence_ids = _collect_evidence_ids(
            evidence_list
        )

        _add_candidate_score(
            scores,
            evidence_map,
            scenario_type="possible_impersonation",
            score=1.5,
            evidence_ids=identity_evidence_ids,
        )

    if signals.get(
        "synthetic_voice_evidence",
        0,
    ) > 0:
        _add_candidate_score(
            scores,
            evidence_map,
            scenario_type="possible_impersonation",
            score=1.5,
            evidence_ids=_collect_evidence_ids(
                evidence_list
            ),
        )

    if signals.get(
        "biometric_mismatch_evidence",
        0,
    ) > 0:
        _add_candidate_score(
            scores,
            evidence_map,
            scenario_type="possible_impersonation",
            score=2.0,
            evidence_ids=_collect_evidence_ids(
                evidence_list
            ),
        )

    if signals.get(
        "claimed_relationships",
        0,
    ) > 0:
        _add_candidate_score(
            scores,
            evidence_map,
            scenario_type="possible_impersonation",
            score=0.75,
            evidence_ids=_collect_evidence_ids(
                evidence_list
            ),
        )

    if signals.get(
        "contradicted_relationships",
        0,
    ) > 0:
        _add_candidate_score(
            scores,
            evidence_map,
            scenario_type="possible_impersonation",
            score=2.0,
            evidence_ids=_collect_evidence_ids(
                evidence_list
            ),
        )

    # -----------------------------------------------------------------------
    # Candidate: possible payment diversion
    # -----------------------------------------------------------------------

    financial_terms = set(
        signals.get(
            "financial_terms",
            [],
        )
    )

    if financial_terms:
        _add_candidate_score(
            scores,
            evidence_map,
            scenario_type="possible_payment_diversion",
            score=1.5,
            evidence_ids=_collect_evidence_ids(
                evidence_list
            ),
        )

    if (
        signals.get(
            "high_risk_relationships",
            0,
        )
        > 0
        and financial_terms
    ):
        _add_candidate_score(
            scores,
            evidence_map,
            scenario_type="possible_payment_diversion",
            score=1.0,
            evidence_ids=_collect_evidence_ids(
                evidence_list
            ),
        )

    if signals.get(
        "pressure_terms",
        [],
    ):
        _add_candidate_score(
            scores,
            evidence_map,
            scenario_type="possible_payment_diversion",
            score=0.75,
            evidence_ids=_collect_evidence_ids(
                evidence_list
            ),
        )

    # -----------------------------------------------------------------------
    # Candidate: possible credential request
    # -----------------------------------------------------------------------

    credential_terms = set(
        signals.get(
            "credential_terms",
            [],
        )
    )

    if credential_terms:
        _add_candidate_score(
            scores,
            evidence_map,
            scenario_type="possible_credential_request",
            score=2.5,
            evidence_ids=_collect_evidence_ids(
                evidence_list
            ),
        )

    if signals.get(
        "pressure_terms",
        [],
    ) and credential_terms:
        _add_candidate_score(
            scores,
            evidence_map,
            scenario_type="possible_credential_request",
            score=0.75,
            evidence_ids=_collect_evidence_ids(
                evidence_list
            ),
        )

    # -----------------------------------------------------------------------
    # Candidate: possible social engineering
    # -----------------------------------------------------------------------

    pressure_terms = set(
        signals.get(
            "pressure_terms",
            [],
        )
    )

    if pressure_terms and (
        identity_terms
        or financial_terms
        or credential_terms
    ):
        _add_candidate_score(
            scores,
            evidence_map,
            scenario_type="possible_social_engineering",
            score=2.0,
            evidence_ids=_collect_evidence_ids(
                evidence_list
            ),
        )

    # -----------------------------------------------------------------------
    # Candidate: evidence inconsistency
    # -----------------------------------------------------------------------

    if signals.get(
        "contradicted_relationships",
        0,
    ) > 0:
        _add_candidate_score(
            scores,
            evidence_map,
            scenario_type="evidence_inconsistency",
            score=2.0,
            evidence_ids=_collect_evidence_ids(
                evidence_list
            ),
        )

    # -----------------------------------------------------------------------
    # Candidate: communication event
    # -----------------------------------------------------------------------

    if event_list:
        _add_candidate_score(
            scores,
            evidence_map,
            scenario_type="communication_event",
            score=0.50,
            evidence_ids=_collect_evidence_ids(
                event_list
            ),
        )

    # -----------------------------------------------------------------------
    # Candidate: possible fraud
    # -----------------------------------------------------------------------

    if (
        financial_terms
        and (
            signals.get(
                "contradicted_relationships",
                0,
            )
            > 0
            or signals.get(
                "synthetic_voice_evidence",
                0,
            )
            > 0
        )
    ):
        _add_candidate_score(
            scores,
            evidence_map,
            scenario_type="possible_fraud",
            score=1.5,
            evidence_ids=_collect_evidence_ids(
                evidence_list
            ),
        )

    return {
        "scores": scores,
        "evidence_map": evidence_map,
    }


# ---------------------------------------------------------------------------
# SCENARIO SELECTION
# ---------------------------------------------------------------------------

def select_scenario(
    *,
    explicit: Optional[Dict[str, Any]],
    candidates: Dict[str, Any],
    signals: Dict[str, Any],
    evidence: Iterable[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Select the most defensible scenario.

    Explicit validated scenario information takes precedence over the
    lightweight candidate scorer.
    """

    if explicit is not None:
        return deepcopy(
            explicit
        )

    scores = candidates.get(
        "scores",
        {},
    )

    evidence_map = candidates.get(
        "evidence_map",
        {},
    )

    if not scores:
        return {
            "scenario_type": "unknown",
            "title": "Scenario not determined",
            "summary": (
                "The available evidence is not sufficient "
                "to reconstruct a specific scenario."
            ),
            "confidence": 0.0,
            "evidence_ids": [],
            "supporting_signals": [],
            "source": "inferred",
        }

    ranked = sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    scenario_type, best_score = ranked[0]

    # Minimum evidence threshold.
    if best_score < 1.0:
        return {
            "scenario_type": "unknown",
            "title": "Scenario not determined",
            "summary": (
                "The available evidence is not sufficient "
                "to reconstruct a specific scenario."
            ),
            "confidence": 0.0,
            "evidence_ids": [],
            "supporting_signals": [],
            "source": "inferred",
        }

    # Normalize raw weights into a bounded confidence indicator.
    #
    # This is intentionally not described as a probability.
    confidence = _clamp(
        best_score / 6.0
    )

    if scenario_type == "possible_impersonation":
        title = "Possible impersonation"

        summary = (
            "The available evidence suggests that a person or actor may "
            "be presenting an identity or relationship that requires "
            "independent verification."
        )

    elif scenario_type == "possible_payment_diversion":
        title = "Possible payment diversion"

        summary = (
            "The evidence indicates a financial request or transaction "
            "that may require additional verification before action."
        )

    elif scenario_type == "possible_credential_request":
        title = "Possible credential request"

        summary = (
            "The communication appears to involve a request for "
            "authentication or account-access information."
        )

    elif scenario_type == "possible_social_engineering":
        title = "Possible social engineering"

        summary = (
            "The communication combines pressure or influence with a "
            "request, identity claim, or other high-impact action."
        )

    elif scenario_type == "possible_fraud":
        title = "Possible fraud pattern"

        summary = (
            "Several evidence signals are consistent with a potentially "
            "deceptive financial scenario."
        )

    elif scenario_type == "evidence_inconsistency":
        title = "Evidence inconsistency"

        summary = (
            "The evidence contains conflicting relationship or identity "
            "signals that require verification."
        )

    elif scenario_type == "communication_event":
        title = "Communication scenario"

        summary = (
            "TrustPath identified a sequence of communication events, "
            "but the current evidence does not support a more specific "
            "scenario classification."
        )

    else:
        title = scenario_type.replace(
            "_",
            " ",
        ).title()

        summary = (
            "TrustPath identified a possible scenario pattern from "
            "the available evidence."
        )

    supporting_signals: List[str] = []

    for key in (
        "identity_terms",
        "financial_terms",
        "credential_terms",
        "pressure_terms",
        "scenario_types",
    ):
        values = signals.get(
            key,
            [],
        )

        if values:
            supporting_signals.extend(
                str(value)
                for value in values
            )

    _append_unique(
        supporting_signals,
        [
            f"claimed relationships: "
            f"{signals.get('claimed_relationships', 0)}",
            f"contradicted relationships: "
            f"{signals.get('contradicted_relationships', 0)}",
            f"synthetic voice evidence: "
            f"{signals.get('synthetic_voice_evidence', 0)}",
        ],
    )

    return {
        "scenario_type": (
            scenario_type
            if scenario_type in SCENARIO_TYPES
            else "unknown"
        ),
        "title": title,
        "summary": summary,
        "confidence": round(
            confidence,
            4,
        ),
        "evidence_ids": list(
            evidence_map.get(
                scenario_type,
                [],
            )
        ),
        "supporting_signals": supporting_signals,
        "source": "inferred",
    }


# ---------------------------------------------------------------------------
# KEY ACTORS / ACTIONS
# ---------------------------------------------------------------------------

def identify_key_actors(
    entities: Iterable[Any],
    relationships: Iterable[Any],
) -> List[Dict[str, Any]]:
    """
    Identify entities participating in scenario relationships.

    The function does not invent actors.
    """

    entity_map: Dict[
        str,
        Dict[str, Any],
    ] = {}

    for entity in entities:
        entity_dict = _as_dict(
            entity
        )

        entity_id = _clean_string(
            entity_dict.get(
                "entity_id"
            )
        )

        if entity_id:
            entity_map[
                entity_id
            ] = entity_dict

    actor_ids: List[str] = []

    for relationship in relationships:
        relationship_dict = _as_dict(
            relationship
        )

        for key in (
            "subject_entity_id",
            "object_entity_id",
        ):
            entity_id = _clean_string(
                relationship_dict.get(
                    key
                )
            )

            if entity_id:
                _append_unique(
                    actor_ids,
                    [entity_id],
                )

    actors: List[Dict[str, Any]] = []

    for entity_id in actor_ids:
        entity = entity_map.get(
            entity_id
        )

        if entity is None:
            # Synthetic role node, such as role_manager.
            if entity_id.startswith(
                "role_"
            ):
                role_name = entity_id[
                    len("role_"):
                ].replace(
                    "_",
                    " ",
                )

                actors.append(
                    {
                        "entity_id": entity_id,
                        "name": role_name,
                        "entity_type": "role",
                        "status": "claimed",
                    }
                )

            continue

        attributes = entity.get(
            "attributes",
            {},
        )

        if not isinstance(
            attributes,
            dict,
        ):
            attributes = {}

        actors.append(
            {
                "entity_id": entity_id,
                "name": entity.get(
                    "name"
                ),
                "entity_type": entity.get(
                    "entity_type"
                ),
                "status": attributes.get(
                    "evidence_status",
                    "unknown",
                ),
                "claimed_role": attributes.get(
                    "claimed_role"
                ),
            }
        )

    return actors


def identify_key_actions(
    *,
    evidence: Iterable[Any],
    events: Iterable[Any],
    relationships: Iterable[Any],
) -> List[Dict[str, Any]]:
    """
    Extract the major actions represented explicitly in the case.
    """

    actions: List[Dict[str, Any]] = []

    for event in events:
        event_dict = _as_dict(
            event
        )

        title = _clean_string(
            event_dict.get(
                "title"
            )
        )

        description = _clean_string(
            event_dict.get(
                "description"
            )
        )

        if title or description:
            actions.append(
                {
                    "type": (
                        _clean_string(
                            event_dict.get(
                                "event_type"
                            )
                        )
                        or "observed"
                    ),
                    "title": (
                        title
                        or "Observed event"
                    ),
                    "description": (
                        description
                        or ""
                    ),
                    "evidence_ids": list(
                        event_dict.get(
                            "evidence_ids",
                            [],
                        )
                        if isinstance(
                            event_dict.get(
                                "evidence_ids",
                                [],
                            ),
                            list,
                        )
                        else []
                    ),
                }
            )

    relation_labels = {
        "requests": "Requested an action",
        "requested": "Requested an action",
        "sent": "Sent communication",
        "received": "Received communication",
        "transferred_to": "Transferred funds or value",
        "paid_to": "Payment relationship identified",
        "issued_by": "Document issuer relationship identified",
        "references": "Document references another item",
        "claims_role": "Role was claimed",
        "claims_identity": "Identity was claimed",
        "associated_with": "Association identified",
        "works_for": "Employment relationship identified",
        "represents": "Representation relationship identified",
    }

    for relationship in relationships:
        relationship_dict = _as_dict(
            relationship
        )

        relation = _clean_string(
            relationship_dict.get(
                "relation"
            )
        )

        if not relation:
            continue

        action_title = relation_labels.get(
            relation,
            relation.replace(
                "_",
                " ",
            ).title(),
        )

        actions.append(
            {
                "type": "relationship",
                "title": action_title,
                "description": (
                    _clean_string(
                        relationship_dict.get(
                            "explanation"
                        )
                    )
                    or ""
                ),
                "evidence_ids": list(
                    relationship_dict.get(
                        "evidence_ids",
                        [],
                    )
                    if isinstance(
                        relationship_dict.get(
                            "evidence_ids",
                            [],
                        ),
                        list,
                    )
                    else []
                ),
            }
        )

    return actions


# ---------------------------------------------------------------------------
# SCENARIO RISK ASSESSMENT
# ---------------------------------------------------------------------------

def determine_scenario_risk(
    *,
    scenario: Dict[str, Any],
    signals: Dict[str, Any],
    relationships: Iterable[Dict[str, Any]],
) -> str:
    """
    Determine a broad scenario risk level.

    This is not an absolute fraud determination.
    """

    scenario_type = scenario.get(
        "scenario_type"
    )

    if scenario_type == "unknown":
        return "UNKNOWN"

    risk_points = 0.0

    # Scenario type contribution.
    if scenario_type in {
        "possible_impersonation",
        "possible_payment_diversion",
        "possible_credential_request",
        "possible_social_engineering",
        "possible_fraud",
    }:
        risk_points += 2.0

    if scenario_type == "evidence_inconsistency":
        risk_points += 1.5

    # Relationship contribution.
    risk_points += (
        float(
            signals.get(
                "contradicted_relationships",
                0,
            )
        )
        * 1.5
    )

    risk_points += (
        float(
            signals.get(
                "synthetic_voice_evidence",
                0,
            )
        )
        * 1.5
    )

    risk_points += (
        float(
            signals.get(
                "biometric_mismatch_evidence",
                0,
            )
        )
        * 1.5
    )

    if signals.get(
        "financial_terms"
    ):
        risk_points += 1.0

    if signals.get(
        "credential_terms"
    ):
        risk_points += 1.0

    if signals.get(
        "pressure_terms"
    ):
        risk_points += 1.0

    if risk_points >= 7.0:
        return "CRITICAL"

    if risk_points >= 5.0:
        return "HIGH"

    if risk_points >= 2.5:
        return "MEDIUM"

    return "LOW"


# ---------------------------------------------------------------------------
# SCENARIO EXPLANATION
# ---------------------------------------------------------------------------

def build_scenario_explanation(
    *,
    scenario: Dict[str, Any],
    signals: Dict[str, Any],
    relationships: Iterable[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Produce an evidence-backed explanation structure.

    This structure is intended for the future Story and Report layers.
    """

    reasons: List[str] = []

    if signals.get(
        "identity_terms"
    ):
        reasons.append(
            "The evidence contains identity or role-related claims."
        )

    if signals.get(
        "financial_terms"
    ):
        reasons.append(
            "The evidence contains financial or transaction-related signals."
        )

    if signals.get(
        "credential_terms"
    ):
        reasons.append(
            "The evidence contains authentication or credential-related signals."
        )

    if signals.get(
        "pressure_terms"
    ):
        reasons.append(
            "The communication contains urgency or pressure indicators."
        )

    if signals.get(
        "contradicted_relationships",
        0,
    ) > 0:
        reasons.append(
            "At least one relationship is explicitly contradicted by the evidence."
        )

    if signals.get(
        "synthetic_voice_evidence",
        0,
    ) > 0:
        reasons.append(
            "Acoustic evidence contains a synthetic-voice indicator."
        )

    if signals.get(
        "biometric_mismatch_evidence",
        0,
    ) > 0:
        reasons.append(
            "Biometric evidence contains a mismatch indicator."
        )

    if not reasons:
        reasons.append(
            "The scenario is based on the available structured evidence."
        )

    relationship_highlights: List[Dict[str, Any]] = []

    for relationship in relationships:
        relationship_dict = _as_dict(
            relationship
        )

        relationship_highlights.append(
            {
                "relationship_id": relationship_dict.get(
                    "relationship_id"
                ),
                "relation": relationship_dict.get(
                    "relation"
                ),
                "status": relationship_dict.get(
                    "status"
                ),
                "confidence": relationship_dict.get(
                    "confidence"
                ),
                "evidence_ids": relationship_dict.get(
                    "evidence_ids",
                    [],
                ),
            }
        )

    return {
        "reasons": reasons,
        "relationship_highlights": relationship_highlights,
        "evidence_ids": list(
            scenario.get(
                "evidence_ids",
                [],
            )
        ),
    }


# ---------------------------------------------------------------------------
# MAIN PUBLIC ENTRY POINT
# ---------------------------------------------------------------------------

def reconstruct_scenario(
    *,
    evidence: Iterable[Any],
    entities: Iterable[Any],
    relationships: Iterable[Any],
    events: Iterable[Any],
) -> Dict[str, Any]:
    """
    Main TrustPath scenario reconstruction entry point.

    Returns:

        {
            "scenario_id": "...",
            "scenario": {...},
            "signals": {...},
            "key_actors": [...],
            "key_actions": [...],
            "risk_level": "...",
            "explanation": {...},
            "traceability": {...}
        }
    """

    evidence_list = [
        _as_dict(item)
        for item in evidence
    ]

    entity_list = [
        _as_dict(item)
        for item in entities
    ]

    relationship_list = [
        _as_dict(item)
        for item in relationships
    ]

    event_list = [
        _as_dict(item)
        for item in events
    ]

    # Step 1: Check for an explicit scenario.
    explicit = _extract_explicit_scenario(
        evidence_list
    )

    # Step 2: Extract transparent signals.
    signals = extract_scenario_signals(
        evidence_list,
        relationship_list,
        event_list,
    )

    # Step 3: Score candidate scenarios.
    candidates = score_scenario_candidates(
        evidence=evidence_list,
        relationships=relationship_list,
        events=event_list,
        signals=signals,
    )

    # Step 4: Select final scenario.
    scenario = select_scenario(
        explicit=explicit,
        candidates=candidates,
        signals=signals,
        evidence=evidence_list,
    )

    # Step 5: Identify actors and actions.
    key_actors = identify_key_actors(
        entity_list,
        relationship_list,
    )

    key_actions = identify_key_actions(
        evidence=evidence_list,
        events=event_list,
        relationships=relationship_list,
    )

    # Step 6: Determine broad risk level.
    risk_level = determine_scenario_risk(
        scenario=scenario,
        signals=signals,
        relationships=relationship_list,
    )

    # Step 7: Build explanation.
    explanation = build_scenario_explanation(
        scenario=scenario,
        signals=signals,
        relationships=relationship_list,
    )

    # Step 8: Build traceability.
    scenario_evidence_ids = set(
        scenario.get(
            "evidence_ids",
            [],
        )
    )

    evidence_ids = {
        _clean_string(
            item.get("evidence_id")
        )
        for item in evidence_list
    }

    evidence_ids.discard(None)

    valid_trace_ids = sorted(
        scenario_evidence_ids.intersection(
            evidence_ids
        )
    )

    traceability = {
        "scenario_evidence_ids": valid_trace_ids,
        "scenario_evidence_count": len(
            valid_trace_ids
        ),
        "total_evidence_count": len(
            evidence_ids
        ),
        "traceable": (
            bool(valid_trace_ids)
            or scenario.get(
                "scenario_type"
            ) == "unknown"
        ),
    }

    # Additional context helpful for downstream story generation.
    context = {
        "scenario_type": scenario.get(
            "scenario_type"
        ),
        "risk_level": risk_level,
        "actor_count": len(
            key_actors
        ),
        "action_count": len(
            key_actions
        ),
        "relationship_count": len(
            relationship_list
        ),
        "event_count": len(
            event_list
        ),
        "evidence_count": len(
            evidence_list
        ),
    }

    return {
        "scenario_id": _new_id(),
        "scenario": {
            "scenario_type": scenario.get(
                "scenario_type",
                "unknown",
            ),
            "title": scenario.get(
                "title",
                "Scenario not determined",
            ),
            "summary": scenario.get(
                "summary",
                "Insufficient evidence to reconstruct a scenario.",
            ),
            "confidence": _safe_float(
                scenario.get(
                    "confidence"
                )
            ) or 0.0,
            "evidence_ids": list(
                scenario.get(
                    "evidence_ids",
                    [],
                )
            ),
            "supporting_signals": list(
                scenario.get(
                    "supporting_signals",
                    [],
                )
            ),
            "source": scenario.get(
                "source",
                "inferred",
            ),
        },
        "signals": signals,
        "key_actors": key_actors,
        "key_actions": key_actions,
        "risk_level": risk_level,
        "explanation": explanation,
        "traceability": traceability,
        "context": context,
    }


# ---------------------------------------------------------------------------
# SCENARIO STORY INPUT
# ---------------------------------------------------------------------------

def build_story_context(
    scenario_result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Convert scenario reconstruction into a compact context object for the
    future story.py module.
    """

    scenario = scenario_result.get(
        "scenario",
        {},
    )

    return {
        "headline": scenario.get(
            "title",
            "Scenario not determined",
        ),
        "summary": scenario.get(
            "summary",
            "",
        ),
        "scenario_type": scenario.get(
            "scenario_type",
            "unknown",
        ),
        "confidence": scenario.get(
            "confidence",
            0.0,
        ),
        "risk_level": scenario_result.get(
            "risk_level",
            "UNKNOWN",
        ),
        "actors": deepcopy(
            scenario_result.get(
                "key_actors",
                [],
            )
        ),
        "actions": deepcopy(
            scenario_result.get(
                "key_actions",
                [],
            )
        ),
        "signals": deepcopy(
            scenario_result.get(
                "signals",
                {},
            )
        ),
        "explanation": deepcopy(
            scenario_result.get(
                "explanation",
                {},
            )
        ),
        "traceability": deepcopy(
            scenario_result.get(
                "traceability",
                {},
            )
        ),
    }


# ---------------------------------------------------------------------------
# VALIDATION
# ---------------------------------------------------------------------------

def validate_scenario_result(
    result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Validate the structure produced by reconstruct_scenario().
    """

    scenario = result.get(
        "scenario",
        {},
    )

    scenario_type = _clean_string(
        scenario.get(
            "scenario_type"
        )
    )

    title = _clean_string(
        scenario.get(
            "title"
        )
    )

    summary = _clean_string(
        scenario.get(
            "summary"
        )
    )

    confidence = _safe_float(
        scenario.get(
            "confidence"
        )
    )

    risk_level = _clean_string(
        result.get(
            "risk_level"
        )
    )

    traceability = result.get(
        "traceability",
        {},
    )

    errors: List[str] = []

    if not scenario_type:
        errors.append(
            "Missing scenario_type."
        )

    if not title:
        errors.append(
            "Missing scenario title."
        )

    if not summary:
        errors.append(
            "Missing scenario summary."
        )

    if confidence is None:
        errors.append(
            "Invalid scenario confidence."
        )

    if risk_level not in SCENARIO_RISK_LEVELS:
        errors.append(
            "Invalid scenario risk level."
        )

    if not isinstance(
        traceability,
        dict,
    ):
        errors.append(
            "Missing traceability object."
        )

    return {
        "valid": not errors,
        "errors": errors,
    }


# ---------------------------------------------------------------------------
# EXPORTS
# ---------------------------------------------------------------------------

__all__ = [
    "SCENARIO_TYPES",
    "SCENARIO_RISK_LEVELS",
    "extract_scenario_signals",
    "score_scenario_candidates",
    "select_scenario",
    "identify_key_actors",
    "identify_key_actions",
    "determine_scenario_risk",
    "build_scenario_explanation",
    "reconstruct_scenario",
    "build_story_context",
    "validate_scenario_result",
]