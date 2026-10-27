"""
TrustPath Relationship Engine

Purpose
-------
Build a conservative relationship graph from normalized TrustPath evidence
and canonical entities.

This module is isolated from all existing/frozen detection modules.

It:
- creates explicitly supported relationships
- creates conservative inferred relationships
- preserves evidence traceability
- supports user role claims
- validates relationship endpoints
- exposes stable public APIs for the TrustPath orchestrator

Public APIs
-----------
build_relationships(...)
build_relationship_graph(...)
validate_relationships(...)
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Relationship constants
# ---------------------------------------------------------------------------

RELATIONSHIP_TYPES = {
    "claims_role",
    "claims_identity",
    "associated_with",
    "represents",
    "works_for",
    "belongs_to",
    "sent",
    "received",
    "sent_by",
    "requested",
    "requests",
    "attached_to",
    "references",
    "contains",
    "issued_by",
    "paid_to",
    "transferred_to",
    "uses",
    "mentions",
    "related_to",
}


RELATIONSHIP_STATUSES = {
    "claimed",
    "inferred",
    "supported",
    "contradicted",
    "verified",
    "unknown",
}


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def _safe_text(value: Any) -> Optional[str]:
    """Return cleaned text or None."""
    if value is None:
        return None

    text = str(value).strip()

    return text if text else None


def _lower(value: Any) -> str:
    """Safely lowercase a value."""
    text = _safe_text(value)

    return text.lower() if text else ""


def _first_non_empty(*values: Any) -> Optional[str]:
    """Return the first non-empty scalar value."""
    for value in values:
        text = _safe_text(value)

        if text:
            return text

    return None


def _unique(values: Iterable[Any]) -> List[Any]:
    """Preserve insertion order while removing duplicates."""
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
    """Normalize a scalar or sequence to a string list."""
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


def _safe_float(value: Any) -> Optional[float]:
    """Convert value to 0..1 float where possible."""
    if value is None:
        return None

    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None

    return max(
        0.0,
        min(
            1.0,
            converted,
        ),
    )


# ---------------------------------------------------------------------------
# Record access helpers
# ---------------------------------------------------------------------------

def _field(
    record: Any,
    *names: str,
) -> Any:
    """
    Read a field from either:
    - dict
    - Pydantic-like object
    """
    if isinstance(record, dict):
        for name in names:
            if name in record:
                return record[name]

        return None

    for name in names:
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

    if callable(model_dump):
        try:
            dumped = model_dump()

            if isinstance(
                dumped,
                dict,
            ):
                for name in names:
                    if name in dumped:
                        return dumped[name]
        except Exception:
            pass

    return None


def _entity_id(
    entity: Any,
) -> Optional[str]:
    return _first_non_empty(
        _field(
            entity,
            "entity_id",
        ),
        _field(
            entity,
            "id",
        ),
        _field(
            entity,
            "object_id",
        ),
    )


def _entity_name(
    entity: Any,
) -> Optional[str]:
    return _first_non_empty(
        _field(
            entity,
            "name",
        ),
        _field(
            entity,
            "display_name",
        ),
        _field(
            entity,
            "value",
        ),
        _field(
            entity,
            "label",
        ),
    )


def _entity_type(
    entity: Any,
) -> str:
    return _lower(
        _first_non_empty(
            _field(
                entity,
                "entity_type",
            ),
            _field(
                entity,
                "type",
            ),
            "unknown",
        )
    )


def _evidence_id(
    evidence: Any,
) -> Optional[str]:
    return _first_non_empty(
        _field(
            evidence,
            "evidence_id",
        ),
        _field(
            evidence,
            "id",
        ),
    )


def _evidence_data(
    evidence: Any,
) -> Dict[str, Any]:
    data = _field(
        evidence,
        "detected_data",
    )

    if isinstance(
        data,
        dict,
    ):
        return data

    # Compatibility with dictionary-style adapter output where structured
    # fields may be stored directly on the evidence dictionary.
    if isinstance(
        evidence,
        dict,
    ):
        return evidence

    return {}


def _relationship_id(
    relationship: Any,
) -> Optional[str]:
    return _first_non_empty(
        _field(
            relationship,
            "relationship_id",
        ),
        _field(
            relationship,
            "id",
        ),
    )


def _relationship_source(
    relationship: Any,
) -> Optional[str]:
    return _first_non_empty(
        _field(
            relationship,
            "source_entity_id",
        ),
        _field(
            relationship,
            "from_entity_id",
        ),
        _field(
            relationship,
            "subject_entity_id",
        ),
        _field(
            relationship,
            "subject_id",
        ),
    )


def _relationship_target(
    relationship: Any,
) -> Optional[str]:
    return _first_non_empty(
        _field(
            relationship,
            "target_entity_id",
        ),
        _field(
            relationship,
            "to_entity_id",
        ),
        _field(
            relationship,
            "object_entity_id",
        ),
        _field(
            relationship,
            "object_id",
        ),
    )


def _relationship_type(
    relationship: Any,
) -> str:
    return _lower(
        _first_non_empty(
            _field(
                relationship,
                "relationship_type",
            ),
            _field(
                relationship,
                "relation",
            ),
            _field(
                relationship,
                "type",
            ),
            "related_to",
        )
    )


def _relationship_status(
    relationship: Any,
) -> str:
    status = _lower(
        _first_non_empty(
            _field(
                relationship,
                "status",
            ),
            "unknown",
        )
    )

    if status not in RELATIONSHIP_STATUSES:
        return "unknown"

    return status


def _relationship_evidence_ids(
    relationship: Any,
) -> List[str]:
    value = _field(
        relationship,
        "evidence_ids",
    )

    if value is None:
        value = _field(
            relationship,
            "evidence_id",
        )

    return _evidence_ids(
        value
    )


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def _normalize_entities(
    entities: Any,
) -> List[Any]:
    if entities is None:
        return []

    if isinstance(
        entities,
        dict,
    ):
        nested = entities.get(
            "entities"
        )

        if isinstance(
            nested,
            list,
        ):
            entities = nested
        else:
            entities = [
                entities
            ]

    if not isinstance(
        entities,
        (list, tuple),
    ):
        return []

    return list(
        entities
    )


def _normalize_evidence(
    evidence: Any,
) -> List[Any]:
    if evidence is None:
        return []

    if isinstance(
        evidence,
        dict,
    ):
        nested = evidence.get(
            "evidence"
        )

        if isinstance(
            nested,
            list,
        ):
            evidence = nested
        else:
            evidence = [
                evidence
            ]

    if not isinstance(
        evidence,
        (list, tuple),
    ):
        return []

    return list(
        evidence
    )


# ---------------------------------------------------------------------------
# Entity lookup
# ---------------------------------------------------------------------------

def _build_entity_lookup(
    entities: Sequence[Any],
) -> Dict[str, Any]:
    lookup: Dict[str, Any] = {}

    for entity in entities:
        entity_id = _entity_id(
            entity
        )

        if entity_id:
            lookup[
                entity_id
            ] = entity

    return lookup


# ---------------------------------------------------------------------------
# Synthetic role nodes
# ---------------------------------------------------------------------------

def _role_node_id(
    role: str,
) -> str:
    """
    Convert a role name into a stable synthetic endpoint.

    Example:
        manager -> role_manager
    """
    normalized = (
        role.strip()
        .lower()
        .replace(
            " ",
            "_",
        )
        .replace(
            "-",
            "_",
        )
    )

    return (
        f"role_{normalized}"
    )


def _is_synthetic_node(
    entity_id: str,
) -> bool:
    return entity_id.startswith(
        "role_"
    )


# ---------------------------------------------------------------------------
# Explicit relationship extraction
# ---------------------------------------------------------------------------

def _build_explicit_relationship(
    *,
    evidence: Any,
    entity_lookup: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """
    Build an explicitly specified relationship.

    Supported keys include:
        source_entity_id
        sender_entity_id
        subject_entity_id

        target_entity_id
        recipient_entity_id
        object_entity_id

        relationship
        relation

        relationship_status
        relationship_confidence
        relationship_explanation
    """
    data = _evidence_data(
        evidence
    )

    source_id = _first_non_empty(
        data.get(
            "source_entity_id"
        ),
        data.get(
            "sender_entity_id"
        ),
        data.get(
            "subject_entity_id"
        ),
        data.get(
            "from_entity_id"
        ),
    )

    target_id = _first_non_empty(
        data.get(
            "target_entity_id"
        ),
        data.get(
            "recipient_entity_id"
        ),
        data.get(
            "object_entity_id"
        ),
        data.get(
            "to_entity_id"
        ),
    )

    relation = _first_non_empty(
        data.get(
            "relationship"
        ),
        data.get(
            "relation"
        ),
        data.get(
            "relationship_type"
        ),
    )

    if not (
        source_id
        and target_id
        and relation
    ):
        return None

    if source_id not in entity_lookup:
        return None

    if target_id not in entity_lookup:
        return None

    normalized_relation = (
        relation.strip()
        .lower()
        .replace(
            " ",
            "_",
        )
        .replace(
            "-",
            "_",
        )
    )

    if normalized_relation not in RELATIONSHIP_TYPES:
        normalized_relation = (
            "related_to"
        )

    status = _lower(
        data.get(
            "relationship_status",
            "supported",
        )
    )

    if status not in RELATIONSHIP_STATUSES:
        status = "unknown"

    return {
        "source_entity_id": source_id,
        "target_entity_id": target_id,
        "relationship_type": normalized_relation,
        "status": status,
        "confidence": _safe_float(
            data.get(
                "relationship_confidence"
            )
        ),
        "evidence_ids": _evidence_ids(
            _evidence_id(evidence)
        ),
        "explanation": _safe_text(
            data.get(
                "relationship_explanation"
            )
        ),
    }


# ---------------------------------------------------------------------------
# Sender / recipient inference
# ---------------------------------------------------------------------------

def _infer_sender_recipient(
    *,
    evidence: Any,
    entity_lookup: Dict[str, Any],
) -> List[Dict[str, Any]]:
    data = _evidence_data(
        evidence
    )

    evidence_id = _evidence_id(
        evidence
    )

    sender_id = _first_non_empty(
        data.get(
            "sender_entity_id"
        ),
        data.get(
            "source_entity_id"
        ),
    )

    recipient_id = _first_non_empty(
        data.get(
            "recipient_entity_id"
        ),
        data.get(
            "target_entity_id"
        ),
    )

    relationships: List[
        Dict[str, Any]
    ] = []

    if (
        sender_id
        and recipient_id
        and sender_id in entity_lookup
        and recipient_id in entity_lookup
    ):
        relationships.append(
            {
                "source_entity_id": sender_id,
                "target_entity_id": recipient_id,
                "relationship_type": "sent",
                "status": "inferred",
                "confidence": 0.70,
                "evidence_ids": _evidence_ids(
                    evidence_id
                ),
                "explanation": (
                    "Inferred from explicit sender and recipient "
                    "metadata in the evidence."
                ),
            }
        )

    return relationships


# ---------------------------------------------------------------------------
# Sender -> organization inference
# ---------------------------------------------------------------------------

def _infer_sender_organization(
    *,
    evidence: Any,
    entity_lookup: Dict[str, Any],
) -> List[Dict[str, Any]]:
    data = _evidence_data(
        evidence
    )

    evidence_id = _evidence_id(
        evidence
    )

    sender_id = _first_non_empty(
        data.get(
            "sender_entity_id"
        ),
        data.get(
            "source_entity_id"
        ),
    )

    organization_id = _first_non_empty(
        data.get(
            "organization_entity_id"
        ),
        data.get(
            "company_entity_id"
        ),
        data.get(
            "sender_organization_id"
        ),
    )

    if not (
        sender_id
        and organization_id
    ):
        return []

    if sender_id not in entity_lookup:
        return []

    if organization_id not in entity_lookup:
        return []

    return [
        {
            "source_entity_id": sender_id,
            "target_entity_id": organization_id,
            "relationship_type": "associated_with",
            "status": "inferred",
            "confidence": 0.65,
            "evidence_ids": _evidence_ids(
                evidence_id
            ),
            "explanation": (
                "Inferred from explicit sender-to-organization "
                "metadata."
            ),
        }
    ]


# ---------------------------------------------------------------------------
# Document -> organization inference
# ---------------------------------------------------------------------------

def _infer_document_organization(
    *,
    evidence: Any,
    entity_lookup: Dict[str, Any],
) -> List[Dict[str, Any]]:
    data = _evidence_data(
        evidence
    )

    evidence_id = _evidence_id(
        evidence
    )

    document_id = _first_non_empty(
        data.get(
            "document_entity_id"
        ),
        data.get(
            "invoice_entity_id"
        ),
    )

    organization_id = _first_non_empty(
        data.get(
            "organization_entity_id"
        ),
        data.get(
            "company_entity_id"
        ),
        data.get(
            "issued_by_entity_id"
        ),
    )

    if not (
        document_id
        and organization_id
    ):
        return []

    if document_id not in entity_lookup:
        return []

    if organization_id not in entity_lookup:
        return []

    return [
        {
            "source_entity_id": document_id,
            "target_entity_id": organization_id,
            "relationship_type": "issued_by",
            "status": "inferred",
            "confidence": 0.70,
            "evidence_ids": _evidence_ids(
                evidence_id
            ),
            "explanation": (
                "Inferred from explicit document issuer metadata."
            ),
        }
    ]


# ---------------------------------------------------------------------------
# Transaction / account relationships
# ---------------------------------------------------------------------------

def _infer_transaction_account(
    *,
    evidence: Any,
    entity_lookup: Dict[str, Any],
) -> List[Dict[str, Any]]:
    data = _evidence_data(
        evidence
    )

    evidence_id = _evidence_id(
        evidence
    )

    transaction_id = _first_non_empty(
        data.get(
            "transaction_entity_id"
        ),
        data.get(
            "transaction_id_entity_id"
        ),
    )

    account_id = _first_non_empty(
        data.get(
            "account_entity_id"
        ),
        data.get(
            "payment_account_entity_id"
        ),
        data.get(
            "destination_account_entity_id"
        ),
    )

    if not (
        transaction_id
        and account_id
    ):
        return []

    if transaction_id not in entity_lookup:
        return []

    if account_id not in entity_lookup:
        return []

    return [
        {
            "source_entity_id": transaction_id,
            "target_entity_id": account_id,
            "relationship_type": "transferred_to",
            "status": "inferred",
            "confidence": 0.72,
            "evidence_ids": _evidence_ids(
                evidence_id
            ),
            "explanation": (
                "Inferred from explicit transaction and account "
                "metadata."
            ),
        }
    ]


# ---------------------------------------------------------------------------
# Invoice -> transaction relationship
# ---------------------------------------------------------------------------

def _infer_invoice_transaction(
    *,
    evidence: Any,
    entity_lookup: Dict[str, Any],
) -> List[Dict[str, Any]]:
    data = _evidence_data(
        evidence
    )

    evidence_id = _evidence_id(
        evidence
    )

    invoice_id = _first_non_empty(
        data.get(
            "invoice_entity_id"
        ),
    )

    transaction_id = _first_non_empty(
        data.get(
            "transaction_entity_id"
        ),
    )

    if not (
        invoice_id
        and transaction_id
    ):
        return []

    if invoice_id not in entity_lookup:
        return []

    if transaction_id not in entity_lookup:
        return []

    return [
        {
            "source_entity_id": invoice_id,
            "target_entity_id": transaction_id,
            "relationship_type": "references",
            "status": "inferred",
            "confidence": 0.70,
            "evidence_ids": _evidence_ids(
                evidence_id
            ),
            "explanation": (
                "Inferred from explicit invoice-to-transaction "
                "metadata."
            ),
        }
    ]


# ---------------------------------------------------------------------------
# User role claims
# ---------------------------------------------------------------------------

def _extract_user_role_claims(
    evidence: Sequence[Any],
    user_context: Any = None,
    entity_lookup: Optional[
        Dict[str, Any]
    ] = None,
) -> List[Dict[str, Any]]:
    """
    Convert explicit user context into claimed relationships.

    Example:
        "Rahul is my manager"

    becomes:

        Rahul -> claims_role -> role_manager

    The relationship is explicitly marked "claimed".
    """
    entity_lookup = (
        entity_lookup
        or {}
    )

    contexts: List[
        str
    ] = []

    if isinstance(
        user_context,
        str,
    ):
        contexts.append(
            user_context
        )

    elif isinstance(
        user_context,
        (list, tuple, set),
    ):
        contexts.extend(
            _string_list(
                user_context
            )
        )

    elif isinstance(
        user_context,
        dict,
    ):
        contexts.extend(
            _string_list(
                user_context.get(
                    "text"
                )
            )
        )

        contexts.extend(
            _string_list(
                user_context.get(
                    "context"
                )
            )
        )

        contexts.extend(
            _string_list(
                user_context.get(
                    "claim"
                )
            )
        )

    # Also inspect evidence-level user context.
    for item in evidence:
        item_context = _field(
            item,
            "user_context",
        )

        contexts.extend(
            _string_list(
                item_context
            )
        )

    contexts = _unique(
        contexts
    )

    relationships: List[
        Dict[str, Any]
    ] = []

    for context in contexts:
        text = context.strip()

        lowered = text.lower()

        # ---------------------------------------------------------------
        # Find a likely person from known entities.
        # ---------------------------------------------------------------

        person_id: Optional[
            str
        ] = None

        for entity_id, entity in entity_lookup.items():
            if (
                _entity_type(entity)
                != "person"
            ):
                continue

            name = _entity_name(
                entity
            )

            if not name:
                continue

            if name.lower() in lowered:
                person_id = entity_id
                break

        if not person_id:
            continue

        # ---------------------------------------------------------------
        # Role extraction
        # ---------------------------------------------------------------

        role: Optional[
            str
        ] = None

        role_keywords = {
            "manager": "manager",
            "boss": "manager",
            "supervisor": "supervisor",
            "director": "director",
            "ceo": "ceo",
            "client": "client",
            "customer": "customer",
            "employee": "employee",
            "colleague": "colleague",
            "friend": "friend",
            "teacher": "teacher",
            "professor": "professor",
            "bank": "bank representative",
            "accountant": "accountant",
        }

        for keyword, normalized_role in role_keywords.items():
            if keyword in lowered:
                role = normalized_role
                break

        if not role:
            continue

        synthetic_role_id = _role_node_id(
            role
        )

        # ---------------------------------------------------------------
        # Evidence IDs related to this context.
        # ---------------------------------------------------------------

        context_evidence_ids: List[
            str
        ] = []

        for item in evidence:
            item_context = _field(
                item,
                "user_context",
            )

            if (
                isinstance(
                    item_context,
                    str,
                )
                and item_context.strip()
                == context
            ):
                evidence_id = _evidence_id(
                    item
                )

                if evidence_id:
                    context_evidence_ids.append(
                        evidence_id
                    )

        relationships.append(
            {
                "source_entity_id": person_id,
                "target_entity_id": synthetic_role_id,
                "relationship_type": "claims_role",
                "status": "claimed",
                "confidence": None,
                "evidence_ids": _unique(
                    context_evidence_ids
                ),
                "explanation": (
                    "Role supplied by the user as contextual information. "
                    "This is a claim and is not independently verified."
                ),
            }
        )

    return relationships


# ---------------------------------------------------------------------------
# Relationship record construction
# ---------------------------------------------------------------------------

def _make_relationship(
    *,
    relationship_id: str,
    candidate: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Produce the canonical dictionary representation.
    """
    source_entity_id = _safe_text(
        candidate.get(
            "source_entity_id"
        )
    )
    target_entity_id = _safe_text(
        candidate.get(
            "target_entity_id"
        )
    )
    relationship_type = (
        _safe_text(
            candidate.get(
                "relationship_type"
            )
        )
        or "related_to"
    )
    status = (
        _safe_text(
            candidate.get(
                "status"
            )
        )
        or "unknown"
    )
    confidence = _safe_float(
        candidate.get(
            "confidence"
        )
    )
    evidence_ids = _evidence_ids(
        candidate.get(
            "evidence_ids"
        )
    )
    explanation = _safe_text(
        candidate.get(
            "explanation"
        )
    )

    # Keep both TrustPath naming conventions so older/newer orchestration
    # layers cannot turn a valid relationship into None -> None.
    return {
        "relationship_id": relationship_id,
        "source_entity_id": source_entity_id,
        "target_entity_id": target_entity_id,
        "relationship_type": relationship_type,
        "subject_entity_id": source_entity_id,
        "object_entity_id": target_entity_id,
        "relation": relationship_type,
        "status": status,
        "confidence": confidence,
        "evidence_ids": evidence_ids,
        "explanation": explanation,
    }


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def _relationship_signature(
    relationship: Dict[str, Any],
) -> Tuple[
    str,
    str,
    str,
]:
    return (
        _safe_text(
            relationship.get(
                "source_entity_id"
            )
        )
        or "",
        _safe_text(
            relationship.get(
                "relationship_type"
            )
        )
        or "",
        _safe_text(
            relationship.get(
                "target_entity_id"
            )
        )
        or "",
    )


def _merge_relationship(
    existing: Dict[str, Any],
    incoming: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Merge duplicate relationships without losing evidence.
    """
    merged = dict(
        existing
    )

    merged["evidence_ids"] = _unique(
        _evidence_ids(
            existing.get(
                "evidence_ids"
            )
        )
        + _evidence_ids(
            incoming.get(
                "evidence_ids"
            )
        )
    )

    existing_confidence = _safe_float(
        existing.get(
            "confidence"
        )
    )

    incoming_confidence = _safe_float(
        incoming.get(
            "confidence"
        )
    )

    if (
        incoming_confidence is not None
        and (
            existing_confidence is None
            or incoming_confidence
            > existing_confidence
        )
    ):
        merged[
            "confidence"
        ] = incoming_confidence

    existing_status = _relationship_status(
        existing
    )

    incoming_status = _relationship_status(
        incoming
    )

    # Conservative status precedence.
    priority = {
        "contradicted": 6,
        "verified": 5,
        "supported": 4,
        "claimed": 3,
        "inferred": 2,
        "unknown": 1,
    }

    if priority.get(
        incoming_status,
        0,
    ) > priority.get(
        existing_status,
        0,
    ):
        merged[
            "status"
        ] = incoming_status

    explanations = _unique(
        [
            _safe_text(
                existing.get(
                    "explanation"
                )
            ),
            _safe_text(
                incoming.get(
                    "explanation"
                )
            ),
        ]
    )

    explanations = [
        value
        for value in explanations
        if value
    ]

    if explanations:
        merged[
            "explanation"
        ] = " ".join(
            explanations
        )

    return merged


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_relationships(
    relationships: Sequence[Any],
    entities: Sequence[Any],
) -> Dict[str, Any]:
    """
    Validate relationship endpoints and evidence traceability.
    """
    normalized_relationships = (
        list(
            relationships
            or []
        )
    )

    normalized_entities = (
        list(
            entities
            or []
        )
    )

    entity_ids = {
        entity_id
        for entity_id in (
            _entity_id(entity)
            for entity in normalized_entities
        )
        if entity_id
    }

    valid_endpoints = 0
    invalid_endpoints = 0
    with_evidence = 0
    without_evidence = 0

    invalid_relationship_ids: List[
        str
    ] = []

    untraced_relationship_ids: List[
        str
    ] = []

    for relationship in normalized_relationships:
        relationship_id = (
            _relationship_id(
                relationship
            )
            or "unknown"
        )

        source_id = (
            _relationship_source(
                relationship
            )
        )

        target_id = (
            _relationship_target(
                relationship
            )
        )

        source_valid = (
            source_id in entity_ids
            or (
                source_id
                and _is_synthetic_node(
                    source_id
                )
            )
        )

        target_valid = (
            target_id in entity_ids
            or (
                target_id
                and _is_synthetic_node(
                    target_id
                )
            )
        )

        if source_valid and target_valid:
            valid_endpoints += 1
        else:
            invalid_endpoints += 1

            invalid_relationship_ids.append(
                relationship_id
            )

        evidence_ids = (
            _relationship_evidence_ids(
                relationship
            )
        )

        if evidence_ids:
            with_evidence += 1
        else:
            without_evidence += 1

            untraced_relationship_ids.append(
                relationship_id
            )

    return {
        "total_relationships": len(
            normalized_relationships
        ),
        "valid_endpoints": valid_endpoints,
        "invalid_endpoints": invalid_endpoints,
        "with_evidence": with_evidence,
        "without_evidence": without_evidence,
        "invalid_relationship_ids": invalid_relationship_ids,
        "untraced_relationship_ids": untraced_relationship_ids,
        "valid": (
            invalid_endpoints == 0
        ),
    }


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_relationships(
    *,
    evidence: Optional[
        Sequence[Any]
    ] = None,
    entities: Optional[
        Sequence[Any]
    ] = None,
    user_context: Any = None,
) -> Dict[str, Any]:
    """
    Build the TrustPath relationship graph.

    This is the primary public API.
    """
    normalized_evidence = _normalize_evidence(
        evidence
    )

    normalized_entities = _normalize_entities(
        entities
    )

    entity_lookup = _build_entity_lookup(
        normalized_entities
    )

    candidates: List[
        Dict[str, Any]
    ] = []

    # ---------------------------------------------------------------
    # User/context claims
    # ---------------------------------------------------------------

    candidates.extend(
        _extract_user_role_claims(
            normalized_evidence,
            user_context=user_context,
            entity_lookup=entity_lookup,
        )
    )

    # ---------------------------------------------------------------
    # Evidence-level relationships
    # ---------------------------------------------------------------

    for item in normalized_evidence:
        explicit = _build_explicit_relationship(
            evidence=item,
            entity_lookup=entity_lookup,
        )

        if explicit:
            candidates.append(
                explicit
            )

        candidates.extend(
            _infer_sender_recipient(
                evidence=item,
                entity_lookup=entity_lookup,
            )
        )

        candidates.extend(
            _infer_sender_organization(
                evidence=item,
                entity_lookup=entity_lookup,
            )
        )

        candidates.extend(
            _infer_document_organization(
                evidence=item,
                entity_lookup=entity_lookup,
            )
        )

        candidates.extend(
            _infer_transaction_account(
                evidence=item,
                entity_lookup=entity_lookup,
            )
        )

        candidates.extend(
            _infer_invoice_transaction(
                evidence=item,
                entity_lookup=entity_lookup,
            )
        )

    # ---------------------------------------------------------------
    # Canonical relationship records
    # ---------------------------------------------------------------

    relationships: List[
        Dict[str, Any]
    ] = []

    by_signature: Dict[
        Tuple[str, str, str],
        Dict[str, Any],
    ] = {}

    for candidate in candidates:
        source_id = _safe_text(
            candidate.get(
                "source_entity_id"
            )
        )

        target_id = _safe_text(
            candidate.get(
                "target_entity_id"
            )
        )

        if not source_id or not target_id:
            continue

        relationship_type = (
            _lower(
                candidate.get(
                    "relationship_type"
                )
            )
            or "related_to"
        )

        if relationship_type not in RELATIONSHIP_TYPES:
            relationship_type = "related_to"

        candidate[
            "relationship_type"
        ] = relationship_type

        signature = (
            source_id,
            relationship_type,
            target_id,
        )

        canonical = _make_relationship(
            relationship_id=(
                f"R{len(relationships) + 1:03d}"
            ),
            candidate=candidate,
        )

        if signature in by_signature:
            merged = _merge_relationship(
                by_signature[
                    signature
                ],
                canonical,
            )

            by_signature[
                signature
            ] = merged

            for index, existing in enumerate(
                relationships
            ):
                if (
                    _relationship_signature(
                        existing
                    )
                    == signature
                ):
                    relationships[
                        index
                    ] = merged
                    break

        else:
            by_signature[
                signature
            ] = canonical

            relationships.append(
                canonical
            )

    # ---------------------------------------------------------------
    # Ensure stable IDs
    # ---------------------------------------------------------------

    for index, relationship in enumerate(
        relationships,
        start=1,
    ):
        relationship[
            "relationship_id"
        ] = f"R{index:03d}"

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------

    by_type: Dict[
        str,
        int,
    ] = {}

    by_status: Dict[
        str,
        int,
    ] = {}

    for relationship in relationships:
        relationship_type = (
            _relationship_type(
                relationship
            )
        )

        status = _relationship_status(
            relationship
        )

        by_type[
            relationship_type
        ] = (
            by_type.get(
                relationship_type,
                0,
            )
            + 1
        )

        by_status[
            status
        ] = (
            by_status.get(
                status,
                0,
            )
            + 1
        )

    # ---------------------------------------------------------------
    # Validation
    # ---------------------------------------------------------------

    validation = validate_relationships(
        relationships=relationships,
        entities=normalized_entities,
    )

    # ---------------------------------------------------------------
    # Traceability
    # ---------------------------------------------------------------

    traceable_count = sum(
        1
        for relationship in relationships
        if _relationship_evidence_ids(
            relationship
        )
    )

    traceability = {
        "total_relationships": len(
            relationships
        ),
        "traceable_relationships": (
            traceable_count
        ),
        "untraceable_relationships": (
            len(relationships)
            - traceable_count
        ),
        "valid": (
            validation[
                "valid"
            ]
            and (
                len(
                    validation[
                        "untraced_relationship_ids"
                    ]
                )
                == 0
            )
        ),
    }

    return {
        "relationships": relationships,
        "summary": {
            "total": len(
                relationships
            ),
            "by_type": by_type,
            "by_status": by_status,
        },
        "validation": validation,
        "traceability": traceability,
        "metadata": {
            "builder": (
                "trustpath.relationships"
            ),
            "deterministic": True,
            "llm_used": False,
            "synthetic_role_nodes_supported": True,
        },
    }


# ---------------------------------------------------------------------------
# Graph alias
# ---------------------------------------------------------------------------

def build_relationship_graph(
    *,
    evidence: Optional[
        Sequence[Any]
    ] = None,
    entities: Optional[
        Sequence[Any]
    ] = None,
    user_context: Any = None,
) -> Dict[str, Any]:
    """
    Compatibility alias for the TrustPath orchestrator.

    Both names intentionally call the same implementation.
    """
    return build_relationships(
        evidence=evidence,
        entities=entities,
        user_context=user_context,
    )


# ---------------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------------

__all__ = [
    "build_relationships",
    "build_relationship_graph",
    "validate_relationships",
    "RELATIONSHIP_TYPES",
    "RELATIONSHIP_STATUSES",
]