"""
TrustPath Contradiction Engine

Purpose
-------
Detect and structure conflicts between TrustPath evidence, entities,
relationships, and events.

This module is deterministic and explainable.

It does NOT:
- modify Text, Visual, Acoustic, Biometric, or Live modules
- verify identity by itself
- use an LLM
- invent evidence
- convert uncertainty into certainty

It produces structured contradiction records with:
- contradiction_id
- contradiction_type
- severity
- description
- evidence_ids
- related entities
- related relationships
- explanation
- traceability
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEVERITY_ORDER = {
    "LOW": 1,
    "MEDIUM": 2,
    "HIGH": 3,
    "CRITICAL": 4,
}

CONTRADICTION_TYPES = {
    "identity_conflict",
    "relationship_conflict",
    "payment_detail_conflict",
    "document_detail_conflict",
    "event_conflict",
    "biometric_identity_conflict",
    "acoustic_identity_conflict",
    "source_conflict",
    "explicit_conflict",
}


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def _safe_text(value: Any) -> Optional[str]:
    if value is None:
        return None

    text = str(value).strip()

    return text if text else None


def _lower(value: Any) -> str:
    text = _safe_text(value)

    return text.lower() if text else ""


def _first_non_empty(*values: Any) -> Optional[str]:
    for value in values:
        text = _safe_text(value)

        if text:
            return text

    return None


def _unique(values: Iterable[Any]) -> List[Any]:
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
    """
    Normalize evidence IDs without converting a list into a string such as:

        "['EV1']"
    """
    return _string_list(value)


def _confidence(value: Any) -> Optional[float]:
    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Input normalization
# ---------------------------------------------------------------------------

def _normalize_records(
    value: Any,
    key: str,
) -> List[Dict[str, Any]]:
    if value is None:
        return []

    if isinstance(value, dict):
        nested = value.get(key)

        if isinstance(nested, list):
            value = nested
        else:
            value = [value]

    if not isinstance(value, (list, tuple)):
        return []

    return [
        dict(item)
        for item in value
        if isinstance(item, dict)
    ]


def _normalize_evidence(
    evidence: Any,
) -> List[Dict[str, Any]]:
    return _normalize_records(
        evidence,
        "evidence",
    )


def _normalize_entities(
    entities: Any,
) -> List[Dict[str, Any]]:
    return _normalize_records(
        entities,
        "entities",
    )


def _normalize_relationships(
    relationships: Any,
) -> List[Dict[str, Any]]:
    return _normalize_records(
        relationships,
        "relationships",
    )


def _normalize_events(
    events: Any,
) -> List[Dict[str, Any]]:
    return _normalize_records(
        events,
        "events",
    )


# ---------------------------------------------------------------------------
# Record helpers
# ---------------------------------------------------------------------------

def _record_id(
    record: Dict[str, Any],
    *keys: str,
) -> Optional[str]:
    values = [
        record.get(key)
        for key in keys
    ]

    return _first_non_empty(*values)


def _entity_id(
    entity: Dict[str, Any],
) -> Optional[str]:
    return _record_id(
        entity,
        "entity_id",
        "id",
        "object_id",
    )


def _entity_name(
    entity: Dict[str, Any],
) -> Optional[str]:
    return _first_non_empty(
        entity.get("name"),
        entity.get("display_name"),
        entity.get("value"),
        entity.get("label"),
    )


def _entity_type(
    entity: Dict[str, Any],
) -> str:
    return _lower(
        _first_non_empty(
            entity.get("entity_type"),
            entity.get("type"),
            "unknown",
        )
    )


def _evidence_id(
    evidence: Dict[str, Any],
) -> Optional[str]:
    return _record_id(
        evidence,
        "evidence_id",
        "id",
    )


def _evidence_source(
    evidence: Dict[str, Any],
) -> str:
    return (
        _first_non_empty(
            evidence.get("source"),
            evidence.get("module"),
            "unknown",
        )
        or "unknown"
    )


def _relationship_source(
    relationship: Dict[str, Any],
) -> Optional[str]:
    return _record_id(
        relationship,
        "source_entity_id",
        "from_entity_id",
        "subject_id",
        "source",
    )


def _relationship_target(
    relationship: Dict[str, Any],
) -> Optional[str]:
    return _record_id(
        relationship,
        "target_entity_id",
        "to_entity_id",
        "object_id",
        "target",
    )


def _relationship_type(
    relationship: Dict[str, Any],
) -> str:
    return _lower(
        _first_non_empty(
            relationship.get("relationship_type"),
            relationship.get("type"),
            "related_to",
        )
    )


def _relationship_status(
    relationship: Dict[str, Any],
) -> str:
    return _lower(
        _first_non_empty(
            relationship.get("status"),
            "unknown",
        )
    )


def _relationship_evidence_ids(
    relationship: Dict[str, Any],
) -> List[str]:
    value = relationship.get("evidence_ids")

    if value is None:
        value = relationship.get("evidence_id")

    return _evidence_ids(value)


# ---------------------------------------------------------------------------
# Lookup builders
# ---------------------------------------------------------------------------

def _build_entity_lookup(
    entities: Sequence[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    lookup: Dict[str, Dict[str, Any]] = {}

    for entity in entities:
        entity_id = _entity_id(entity)

        if entity_id:
            lookup[entity_id] = entity

    return lookup


def _build_evidence_lookup(
    evidence: Sequence[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    lookup: Dict[str, Dict[str, Any]] = {}

    for item in evidence:
        evidence_id = _evidence_id(item)

        if evidence_id:
            lookup[evidence_id] = item

    return lookup


def _build_entity_name_index(
    entities: Sequence[Dict[str, Any]],
) -> Dict[Tuple[str, str], List[str]]:
    """
    Build normalized (type, name) -> entity IDs.
    """
    index: Dict[Tuple[str, str], List[str]] = {}

    for entity in entities:
        entity_id = _entity_id(entity)
        entity_name = _entity_name(entity)
        entity_type = _entity_type(entity)

        if not entity_id or not entity_name:
            continue

        key = (
            entity_type,
            entity_name.strip().lower(),
        )

        index.setdefault(
            key,
            [],
        ).append(entity_id)

    return index


# ---------------------------------------------------------------------------
# Text/value extraction
# ---------------------------------------------------------------------------

def _record_text(
    record: Dict[str, Any],
) -> str:
    values: List[str] = []

    for key in (
        "text",
        "content",
        "summary",
        "description",
        "message",
        "transcript",
        "ocr_text",
        "classification",
        "visual_classification",
        "intent",
        "action",
        "event_type",
        "type",
    ):
        text = _safe_text(
            record.get(key)
        )

        if text:
            values.append(text)

    return " ".join(values)


def _explicit_contradictions(
    evidence: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Pick up contradiction fields already supplied upstream.

    Supported examples:
        contradiction=True
        contradictions=[...]
        conflict=True
        conflict_description="..."
    """
    results: List[Dict[str, Any]] = []

    for item in evidence:
        contradiction_flag = (
            item.get("contradiction") is True
            or item.get("conflict") is True
        )

        descriptions = _string_list(
            item.get("contradictions")
        )

        descriptions.extend(
            _string_list(
                item.get("conflicts")
            )
        )

        description = _first_non_empty(
            item.get("contradiction_description"),
            item.get("conflict_description"),
        )

        if description:
            descriptions.append(
                description
            )

        if contradiction_flag and not descriptions:
            descriptions.append(
                "The evidence record explicitly marks a conflict."
            )

        evidence_id = _evidence_id(item)

        if not evidence_id:
            continue

        for description_text in _unique(
            descriptions
        ):
            results.append(
                {
                    "type": "explicit_conflict",
                    "severity": _safe_text(
                        item.get("contradiction_severity")
                    ) or "MEDIUM",
                    "description": description_text,
                    "evidence_ids": [evidence_id],
                }
            )

    return results


# ---------------------------------------------------------------------------
# Identity contradiction detection
# ---------------------------------------------------------------------------

def _detect_biometric_identity_conflict(
    evidence: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Detect an explicit biometric mismatch.

    We only treat it as a contradiction when the biometric module itself
    supplies a mismatch/non-match indication.
    """
    biometric_records = [
        item
        for item in evidence
        if _evidence_source(item).lower()
        == "biometric"
    ]

    results: List[Dict[str, Any]] = []

    for item in biometric_records:
        evidence_id = _evidence_id(item)

        if not evidence_id:
            continue

        mismatch = (
            item.get("match") is False
            or item.get("matched") is False
            or item.get("identity_match") is False
            or item.get("verification_passed") is False
        )

        classification = _lower(
            _first_non_empty(
                item.get("classification"),
                item.get("result"),
                item.get("status"),
            )
        )

        if any(
            term in classification
            for term in (
                "mismatch",
                "non_match",
                "not_match",
                "different",
                "failed",
                "unknown",
            )
        ):
            mismatch = True

        if not mismatch:
            continue

        results.append(
            {
                "type": "biometric_identity_conflict",
                "severity": "HIGH",
                "description": (
                    "The biometric evidence indicates that the "
                    "captured speaker may not match the claimed "
                    "identity."
                ),
                "evidence_ids": [
                    evidence_id
                ],
            }
        )

    return results


# ---------------------------------------------------------------------------
# Acoustic contradiction detection
# ---------------------------------------------------------------------------

def _detect_acoustic_identity_conflict(
    evidence: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Identify acoustic evidence that conflicts with an identity claim.

    Synthetic voice alone is not treated as proof of impersonation.
    The contradiction is phrased conservatively.
    """
    acoustic_records = [
        item
        for item in evidence
        if _evidence_source(item).lower()
        == "acoustic"
    ]

    results: List[Dict[str, Any]] = []

    for item in acoustic_records:
        evidence_id = _evidence_id(item)

        if not evidence_id:
            continue

        synthetic = (
            item.get("synthetic_detected") is True
        )

        classification = _lower(
            _first_non_empty(
                item.get("classification"),
                item.get("result"),
            )
        )

        if any(
            term in classification
            for term in (
                "synthetic",
                "generated",
                "spoof",
                "fake",
            )
        ):
            synthetic = True

        if not synthetic:
            continue

        results.append(
            {
                "type": "acoustic_identity_conflict",
                "severity": "HIGH",
                "description": (
                    "The acoustic evidence indicates possible "
                    "synthetic or manipulated speech, which can "
                    "conflict with a claimed real-world identity."
                ),
                "evidence_ids": [
                    evidence_id
                ],
            }
        )

    return results


# ---------------------------------------------------------------------------
# Relationship conflict detection
# ---------------------------------------------------------------------------

def _relationship_key(
    relationship: Dict[str, Any],
) -> Tuple[str, str, str]:
    source = (
        _relationship_source(
            relationship
        )
        or ""
    )

    target = (
        _relationship_target(
            relationship
        )
        or ""
    )

    rel_type = _relationship_type(
        relationship
    )

    return (
        source,
        rel_type,
        target,
    )


def _detect_relationship_conflicts(
    relationships: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Detect direct contradictions such as:

        A -> works_for -> X
        A -> works_for -> Y

    or:

        A -> claims_role -> manager
        A -> claims_role -> customer
    """
    grouped: Dict[
        Tuple[str, str],
        List[Dict[str, Any]]
    ] = {}

    for relationship in relationships:
        source = _relationship_source(
            relationship
        )

        rel_type = _relationship_type(
            relationship
        )

        if not source:
            continue

        if rel_type not in {
            "works_for",
            "claims_role",
            "claims_identity",
            "belongs_to",
        }:
            continue

        grouped.setdefault(
            (source, rel_type),
            [],
        ).append(
            relationship
        )

    results: List[Dict[str, Any]] = []

    for (
        source,
        rel_type,
    ), records in grouped.items():
        targets = [
            _relationship_target(
                record
            )
            for record in records
        ]

        targets = [
            target
            for target in targets
            if target
        ]

        unique_targets = _unique(
            targets
        )

        if len(unique_targets) <= 1:
            continue

        evidence_ids: List[str] = []

        for record in records:
            evidence_ids.extend(
                _relationship_evidence_ids(
                    record
                )
            )

        results.append(
            {
                "type": "relationship_conflict",
                "severity": "HIGH",
                "description": (
                    f"The evidence contains conflicting "
                    f"{rel_type.replace('_', ' ')} relationships "
                    f"for the same actor."
                ),
                "evidence_ids": _unique(
                    evidence_ids
                ),
                "source_entity_id": source,
                "target_entity_ids": unique_targets,
            }
        )

    return results


# ---------------------------------------------------------------------------
# Payment/document conflict detection
# ---------------------------------------------------------------------------

def _normalized_value(
    value: Any,
) -> Optional[str]:
    text = _safe_text(value)

    if not text:
        return None

    return (
        text
        .lower()
        .replace(" ", "")
        .replace("-", "")
        .replace("_", "")
    )


def _extract_field(
    record: Dict[str, Any],
    field_names: Sequence[str],
) -> Optional[str]:
    for field_name in field_names:
        value = record.get(
            field_name
        )

        text = _safe_text(
            value
        )

        if text:
            return text

    return None


def _detect_structured_value_conflicts(
    evidence: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Detect explicit conflicting structured payment/document fields.

    Examples:
        account_number
        destination_account
        payment_account
        invoice_number
        transaction_id
        amount
        currency
    """
    field_groups = {
        "payment_account": (
            "account_number",
            "destination_account",
            "destination_account_number",
            "payment_account",
            "payment_account_number",
            "beneficiary_account",
        ),
        "transaction_id": (
            "transaction_id",
            "transaction_number",
            "txn_id",
        ),
        "invoice_number": (
            "invoice_number",
            "invoice_id",
        ),
        "amount": (
            "amount",
            "payment_amount",
            "transaction_amount",
            "invoice_amount",
        ),
        "currency": (
            "currency",
        ),
    }

    results: List[Dict[str, Any]] = []

    for logical_field, field_names in field_groups.items():
        values: Dict[
            str,
            List[str]
        ] = {}

        for item in evidence:
            value = _extract_field(
                item,
                field_names,
            )

            evidence_id = _evidence_id(
                item
            )

            if not value or not evidence_id:
                continue

            normalized = _normalized_value(
                value
            )

            if not normalized:
                continue

            values.setdefault(
                normalized,
                [],
            ).append(
                evidence_id
            )

        if len(values) <= 1:
            continue

        evidence_ids: List[str] = []

        for ids in values.values():
            evidence_ids.extend(
                ids
            )

        results.append(
            {
                "type": (
                    "payment_detail_conflict"
                    if logical_field
                    in {
                        "payment_account",
                        "amount",
                        "currency",
                    }
                    else "document_detail_conflict"
                ),
                "severity": (
                    "HIGH"
                    if logical_field
                    in {
                        "payment_account",
                        "amount",
                    }
                    else "MEDIUM"
                ),
                "description": (
                    f"Different values were found for "
                    f"{logical_field.replace('_', ' ')} "
                    f"across the supplied evidence."
                ),
                "evidence_ids": _unique(
                    evidence_ids
                ),
                "field": logical_field,
            }
        )

    return results


# ---------------------------------------------------------------------------
# Event conflict detection
# ---------------------------------------------------------------------------

def _event_key(
    event: Dict[str, Any],
) -> Optional[str]:
    return _first_non_empty(
        event.get("event_type"),
        event.get("type"),
        event.get("name"),
    )


def _event_description(
    event: Dict[str, Any],
) -> str:
    return (
        _first_non_empty(
            event.get("description"),
            event.get("summary"),
            event.get("action"),
        )
        or ""
    )


def _detect_event_conflicts(
    events: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Detect explicit positive/negative event contradictions.

    This is intentionally conservative and only operates on explicit
    negation patterns.
    """
    groups: Dict[
        str,
        List[Dict[str, Any]]
    ] = {}

    for event in events:
        key = _event_key(event)

        if key:
            groups.setdefault(
                key.lower(),
                [],
            ).append(
                event
            )

    results: List[Dict[str, Any]] = []

    for event_type, records in groups.items():
        positive: List[
            Dict[str, Any]
        ] = []

        negative: List[
            Dict[str, Any]
        ] = []

        for event in records:
            description = _lower(
                _event_description(event)
            )

            if any(
                term in description
                for term in (
                    "not happened",
                    "did not happen",
                    "never happened",
                    "no payment",
                    "wasn't",
                    "was not",
                    "false",
                )
            ):
                negative.append(
                    event
                )
            else:
                positive.append(
                    event
                )

        if not positive or not negative:
            continue

        evidence_ids: List[str] = []

        for event in (
            positive + negative
        ):
            evidence_ids.extend(
                _evidence_ids(
                    event.get(
                        "evidence_ids"
                    )
                )
            )

            if event.get(
                "evidence_ids"
            ) is None:
                evidence_ids.extend(
                    _evidence_ids(
                        event.get(
                            "evidence_id"
                        )
                    )
                )

        results.append(
            {
                "type": "event_conflict",
                "severity": "MEDIUM",
                "description": (
                    f"The supplied events contain conflicting "
                    f"descriptions for '{event_type.replace('_', ' ')}'."
                ),
                "evidence_ids": _unique(
                    evidence_ids
                ),
            }
        )

    return results


# ---------------------------------------------------------------------------
# Contradiction record construction
# ---------------------------------------------------------------------------

def _build_contradiction(
    contradiction_id: str,
    candidate: Dict[str, Any],
    evidence_lookup: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    evidence_ids = _evidence_ids(
        candidate.get(
            "evidence_ids"
        )
    )

    existing_ids = set(
        evidence_lookup.keys()
    )

    valid_evidence_ids = [
        evidence_id
        for evidence_id in evidence_ids
        if evidence_id in existing_ids
    ]

    source_names = [
        _evidence_source(
            evidence_lookup[evidence_id]
        )
        for evidence_id
        in valid_evidence_ids
    ]

    severity = _safe_text(
        candidate.get(
            "severity"
        )
    )

    if not severity:
        severity = "MEDIUM"

    severity = severity.upper()

    if severity not in SEVERITY_ORDER:
        severity = "MEDIUM"

    return {
        "contradiction_id": contradiction_id,
        "contradiction_type": (
            _safe_text(
                candidate.get(
                    "type"
                )
            )
            or "explicit_conflict"
        ),
        "severity": severity,
        "description": (
            _safe_text(
                candidate.get(
                    "description"
                )
            )
            or "Conflicting evidence was identified."
        ),
        "evidence_ids": _unique(
            valid_evidence_ids
        ),
        "sources": _unique(
            source_names
        ),
        "related_entities": _unique(
            _string_list(
                candidate.get(
                    "related_entities"
                )
            )
        ),
        "related_relationships": _unique(
            _string_list(
                candidate.get(
                    "related_relationships"
                )
            )
        ),
        "field": _safe_text(
            candidate.get(
                "field"
            )
        ),
        "explanation": (
            "This contradiction is based only on the supplied "
            "structured evidence. It does not independently establish "
            "which source is truthful."
        ),
    }


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def _contradiction_signature(
    contradiction: Dict[str, Any],
) -> Tuple[
    str,
    str,
    Tuple[str, ...],
]:
    return (
        _safe_text(
            contradiction.get(
                "contradiction_type"
            )
        )
        or "",
        _safe_text(
            contradiction.get(
                "description"
            )
        )
        or "",
        tuple(
            _evidence_ids(
                contradiction.get(
                    "evidence_ids"
                )
            )
        ),
    )


def _deduplicate_contradictions(
    contradictions: Sequence[
        Dict[str, Any]
    ],
) -> List[
    Dict[str, Any]
]:
    result: List[
        Dict[str, Any]
    ] = []

    seen = set()

    for contradiction in contradictions:
        signature = _contradiction_signature(
            contradiction
        )

        if signature in seen:
            continue

        seen.add(
            signature
        )

        result.append(
            contradiction
        )

    return result


# ---------------------------------------------------------------------------
# Traceability
# ---------------------------------------------------------------------------

def _validate_traceability(
    contradictions: Sequence[
        Dict[str, Any]
    ],
    evidence: Sequence[
        Dict[str, Any]
    ],
) -> Dict[str, Any]:
    known_ids = {
        evidence_id
        for evidence_id in (
            _evidence_id(item)
            for item in evidence
        )
        if evidence_id
    }

    referenced_ids: List[str] = []

    for contradiction in contradictions:
        referenced_ids.extend(
            _evidence_ids(
                contradiction.get(
                    "evidence_ids"
                )
            )
        )

    referenced_ids = _unique(
        referenced_ids
    )

    invalid_ids = [
        evidence_id
        for evidence_id in referenced_ids
        if evidence_id not in known_ids
    ]

    return {
        "valid": len(invalid_ids) == 0,
        "referenced_evidence_ids": referenced_ids,
        "invalid_evidence_ids": invalid_ids,
        "total_referenced": len(
            referenced_ids
        ),
        "total_known": len(
            known_ids
        ),
    }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_contradictions(
    result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Validate the contradiction result structure and traceability.
    """
    if not isinstance(
        result,
        dict,
    ):
        return {
            "valid": False,
            "errors": [
                "Contradiction result must be a dictionary."
            ],
        }

    errors: List[str] = []

    required_fields = (
        "contradictions",
        "summary",
        "traceability",
    )

    for field in required_fields:
        if field not in result:
            errors.append(
                f"Missing required field: {field}"
            )

    contradictions = result.get(
        "contradictions"
    )

    if not isinstance(
        contradictions,
        list,
    ):
        errors.append(
            "contradictions must be a list."
        )

    traceability = result.get(
        "traceability"
    )

    if not isinstance(
        traceability,
        dict,
    ):
        errors.append(
            "traceability must be a dictionary."
        )
    elif traceability.get(
        "valid"
    ) is False:
        errors.append(
            "Contradiction evidence references are invalid."
        )

    return {
        "valid": len(errors) == 0,
        "errors": errors,
    }


# ---------------------------------------------------------------------------
# Main contradiction engine
# ---------------------------------------------------------------------------

def detect_contradictions(
    *,
    evidence: Optional[
        Sequence[Dict[str, Any]]
    ] = None,
    entities: Optional[
        Sequence[Dict[str, Any]]
    ] = None,
    relationships: Optional[
        Sequence[Dict[str, Any]]
    ] = None,
    events: Optional[
        Sequence[Dict[str, Any]]
    ] = None,
) -> Dict[str, Any]:
    """
    Detect contradictions across TrustPath evidence layers.
    """
    normalized_evidence = _normalize_evidence(
        evidence
    )

    normalized_entities = _normalize_entities(
        entities
    )

    normalized_relationships = _normalize_relationships(
        relationships
    )

    normalized_events = _normalize_events(
        events
    )

    entity_lookup = _build_entity_lookup(
        normalized_entities
    )

    evidence_lookup = _build_evidence_lookup(
        normalized_evidence
    )

    candidates: List[
        Dict[str, Any]
    ] = []

    # Explicit contradictions supplied upstream.
    candidates.extend(
        _explicit_contradictions(
            normalized_evidence
        )
    )

    # Identity/audio contradictions.
    candidates.extend(
        _detect_biometric_identity_conflict(
            normalized_evidence
        )
    )

    candidates.extend(
        _detect_acoustic_identity_conflict(
            normalized_evidence
        )
    )

    # Relationship contradictions.
    relationship_candidates = (
        _detect_relationship_conflicts(
            normalized_relationships
        )
    )

    candidates.extend(
        relationship_candidates
    )

    # Structured field contradictions.
    candidates.extend(
        _detect_structured_value_conflicts(
            normalized_evidence
        )
    )

    # Event contradictions.
    candidates.extend(
        _detect_event_conflicts(
            normalized_events
        )
    )

    # Add relationship/entity traceability where available.
    for candidate in candidates:
        if (
            "source_entity_id"
            in candidate
        ):
            source_id = candidate.get(
                "source_entity_id"
            )

            if source_id in entity_lookup:
                candidate.setdefault(
                    "related_entities",
                    [],
                ).append(
                    source_id
                )

        candidate["related_entities"] = _unique(
            _string_list(
                candidate.get(
                    "related_entities"
                )
            )
        )

    candidates = _deduplicate_contradictions(
        candidates
    )

    contradictions: List[
        Dict[str, Any]
    ] = []

    for index, candidate in enumerate(
        candidates,
        start=1,
    ):
        contradiction_id = (
            f"CON_{index:03d}"
        )

        contradictions.append(
            _build_contradiction(
                contradiction_id,
                candidate,
                evidence_lookup,
            )
        )

    # Highest severity first.
    contradictions.sort(
        key=lambda item: SEVERITY_ORDER.get(
            item.get(
                "severity",
                "MEDIUM",
            ),
            2,
        ),
        reverse=True,
    )

    # Renumber after sorting.
    for index, contradiction in enumerate(
        contradictions,
        start=1,
    ):
        contradiction[
            "contradiction_id"
        ] = f"CON_{index:03d}"

    by_type: Dict[str, int] = {}
    by_severity: Dict[str, int] = {}

    for contradiction in contradictions:
        contradiction_type = (
            contradiction.get(
                "contradiction_type"
            )
            or "unknown"
        )

        severity = (
            contradiction.get(
                "severity"
            )
            or "MEDIUM"
        )

        by_type[
            contradiction_type
        ] = (
            by_type.get(
                contradiction_type,
                0,
            )
            + 1
        )

        by_severity[
            severity
        ] = (
            by_severity.get(
                severity,
                0,
            )
            + 1
        )

    traceability = _validate_traceability(
        contradictions,
        normalized_evidence,
    )

    result: Dict[str, Any] = {
        "contradictions": contradictions,
        "summary": {
            "total": len(
                contradictions
            ),
            "by_type": by_type,
            "by_severity": by_severity,
            "has_contradictions": bool(
                contradictions
            ),
            "highest_severity": (
                contradictions[0]["severity"]
                if contradictions
                else None
            ),
        },
        "traceability": traceability,
        "metadata": {
            "builder": (
                "trustpath.contradictions"
            ),
            "deterministic": True,
            "llm_used": False,
            "independent_truth_verification": False,
        },
    }

    return result


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def build_contradiction_report(
    reasoning: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Convenience wrapper for an orchestrator-style reasoning object.
    """
    if not isinstance(
        reasoning,
        dict,
    ):
        reasoning = {}

    return detect_contradictions(
        evidence=reasoning.get(
            "evidence"
        ),
        entities=reasoning.get(
            "entities"
        ),
        relationships=reasoning.get(
            "relationships"
        ),
        events=reasoning.get(
            "events"
        ),
    )


__all__ = [
    "detect_contradictions",
    "build_contradiction_report",
    "validate_contradictions",
]