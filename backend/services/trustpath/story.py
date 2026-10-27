"""
TrustPath Story Builder

Transforms structured TrustPath reasoning outputs into a deterministic,
human-readable, evidence-traceable story.

This module:
- does not use an LLM
- does not modify existing detection modules
- does not verify user claims automatically
- does not present predictions as guaranteed facts
- preserves evidence IDs for reporting and auditing
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RISK_LEVELS = {"LOW", "MEDIUM", "HIGH", "CRITICAL"}

DISPLAY_SOURCE_NAMES = {
    "text": "Text",
    "visual": "Visual",
    "acoustic": "Acoustic",
    "biometric": "Biometric",
    "user_context": "User Context",
    "system": "TrustPath",
}


FINANCIAL_TERMS = (
    "payment",
    "pay",
    "transfer",
    "invoice",
    "transaction",
    "bank",
    "account",
    "money",
    "amount",
    "refund",
    "wire",
    "upi",
    "wallet",
)


CREDENTIAL_TERMS = (
    "password",
    "otp",
    "one-time password",
    "login",
    "credential",
    "pin",
    "verification code",
    "passcode",
    "authentication",
)


PRESSURE_TERMS = (
    "urgent",
    "immediately",
    "now",
    "deadline",
    "asap",
    "quickly",
    "confidential",
    "don't tell",
    "do not tell",
)


DOCUMENT_TERMS = (
    "invoice",
    "receipt",
    "bill",
    "document",
    "payment request",
    "statement",
)


# ---------------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------------

def _safe_text(value: Any) -> Optional[str]:
    """Return a cleaned string or None."""
    if value is None:
        return None

    text = str(value).strip()

    return text if text else None


def _lower(value: Any) -> str:
    """Safely convert a value to lowercase text."""
    text = _safe_text(value)

    return text.lower() if text else ""


def _first_non_empty(*values: Any) -> Optional[str]:
    """
    Return the first non-empty textual value.

    Important:
    This function is for scalar values only.
    Do NOT use it for list-valued fields such as evidence_ids.
    """
    for value in values:
        text = _safe_text(value)

        if text:
            return text

    return None


def _unique(values: Iterable[Any]) -> List[Any]:
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


def _string_list(values: Any) -> List[str]:
    """
    Convert a scalar/list/set/tuple into a list of strings.
    """
    if values is None:
        return []

    if isinstance(values, str):
        return [values.strip()] if values.strip() else []

    if isinstance(values, (list, tuple, set)):
        result: List[str] = []

        for value in values:
            text = _safe_text(value)

            if text:
                result.append(text)

        return _unique(result)

    text = _safe_text(values)

    return [text] if text else []


def _evidence_ids(value: Any) -> List[str]:
    """
    Normalize evidence IDs correctly.

    This function is deliberately separate from _first_non_empty()
    because evidence_ids are commonly lists.
    """
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


def _risk_level(value: Any) -> str:
    """Normalize a risk value."""
    risk = _safe_text(value)

    if not risk:
        return "UNKNOWN"

    risk = risk.upper()

    return risk if risk in RISK_LEVELS else risk


def _source_label(source: Any) -> str:
    """Convert internal source name to readable source name."""
    source_text = _safe_text(source)

    if not source_text:
        return "Evidence"

    return DISPLAY_SOURCE_NAMES.get(
        source_text.lower(),
        source_text,
    )


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

def _normalize_evidence(
    evidence: Any,
) -> List[Dict[str, Any]]:
    """Normalize evidence records."""
    if evidence is None:
        return []

    if isinstance(evidence, dict):
        evidence = [evidence]

    if not isinstance(evidence, (list, tuple)):
        return []

    return [
        dict(item)
        for item in evidence
        if isinstance(item, dict)
    ]


def _normalize_entities(
    entities: Any,
) -> List[Dict[str, Any]]:
    """Normalize entities."""
    if entities is None:
        return []

    if isinstance(entities, dict):
        if isinstance(entities.get("entities"), list):
            entities = entities["entities"]
        else:
            entities = [entities]

    if not isinstance(entities, (list, tuple)):
        return []

    return [
        dict(item)
        for item in entities
        if isinstance(item, dict)
    ]


def _normalize_relationships(
    relationships: Any,
) -> List[Dict[str, Any]]:
    """Normalize relationships."""
    if relationships is None:
        return []

    if isinstance(relationships, dict):
        if isinstance(relationships.get("relationships"), list):
            relationships = relationships["relationships"]
        else:
            relationships = [relationships]

    if not isinstance(relationships, (list, tuple)):
        return []

    return [
        dict(item)
        for item in relationships
        if isinstance(item, dict)
    ]


def _normalize_events(
    events: Any,
) -> List[Dict[str, Any]]:
    """Normalize events."""
    if events is None:
        return []

    if isinstance(events, dict):
        if isinstance(events.get("events"), list):
            events = events["events"]
        else:
            events = [events]

    if not isinstance(events, (list, tuple)):
        return []

    return [
        dict(item)
        for item in events
        if isinstance(item, dict)
    ]


def _normalize_contradictions(
    contradictions: Any,
) -> List[Dict[str, Any]]:
    """Normalize contradictions."""
    if contradictions is None:
        return []

    if isinstance(contradictions, dict):
        if isinstance(contradictions.get("contradictions"), list):
            contradictions = contradictions["contradictions"]
        else:
            contradictions = [contradictions]

    if not isinstance(contradictions, (list, tuple)):
        return []

    return [
        dict(item)
        for item in contradictions
        if isinstance(item, dict)
    ]


# ---------------------------------------------------------------------------
# Entity helpers
# ---------------------------------------------------------------------------

def _entity_id(
    entity: Dict[str, Any],
) -> Optional[str]:
    return _first_non_empty(
        entity.get("entity_id"),
        entity.get("id"),
        entity.get("object_id"),
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


def _build_entity_lookup(
    entities: Sequence[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Build entity_id -> entity."""
    lookup: Dict[str, Dict[str, Any]] = {}

    for entity in entities:
        entity_id = _entity_id(entity)

        if entity_id:
            lookup[entity_id] = entity

    return lookup


def _entity_display(
    entity_id: Any,
    entity_lookup: Dict[str, Dict[str, Any]],
) -> Optional[str]:
    """Resolve an entity ID into readable text."""
    entity_id_text = _safe_text(entity_id)

    if not entity_id_text:
        return None

    entity = entity_lookup.get(entity_id_text)

    if entity:
        name = _entity_name(entity)

        if name:
            return name

    if entity_id_text.startswith("role_"):
        role_name = entity_id_text[
            len("role_"):
        ].replace("_", " ").strip()

        if role_name:
            return role_name.title()

    return entity_id_text


# ---------------------------------------------------------------------------
# Evidence helpers
# ---------------------------------------------------------------------------

def _evidence_id(
    evidence: Dict[str, Any],
) -> Optional[str]:
    return _first_non_empty(
        evidence.get("evidence_id"),
        evidence.get("id"),
    )


def _evidence_source(
    evidence: Dict[str, Any],
) -> str:
    return (
        _first_non_empty(
            evidence.get("source"),
            evidence.get("module"),
            "evidence",
        )
        or "evidence"
    )


def _evidence_classification(
    evidence: Dict[str, Any],
) -> Optional[str]:
    return _first_non_empty(
        evidence.get("classification"),
        evidence.get("visual_classification"),
        evidence.get("label"),
        evidence.get("type"),
    )


def _evidence_confidence(
    evidence: Dict[str, Any],
) -> Optional[float]:
    raw = (
        evidence.get("confidence")
        if evidence.get("confidence") is not None
        else evidence.get("score")
    )

    if raw is None:
        return None

    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _evidence_excerpt(
    evidence: Dict[str, Any],
) -> Optional[str]:
    """
    Extract a concise description of the evidence.
    """
    source = _evidence_source(evidence)

    candidates = (
        evidence.get("summary"),
        evidence.get("description"),
        evidence.get("content"),
        evidence.get("text"),
        evidence.get("message"),
        evidence.get("transcript"),
        evidence.get("ocr_text"),
        evidence.get("classification"),
        evidence.get("visual_classification"),
    )

    text = _first_non_empty(*candidates)

    if not text:
        return None

    if len(text) > 280:
        text = text[:277].rstrip() + "..."

    return f"{_source_label(source)} evidence: {text}"


# ---------------------------------------------------------------------------
# Relationship helpers
# ---------------------------------------------------------------------------

def _relationship_source(
    relationship: Dict[str, Any],
) -> Optional[str]:
    return _first_non_empty(
        relationship.get("source_entity_id"),
        relationship.get("from_entity_id"),
        relationship.get("subject_id"),
        relationship.get("source"),
    )


def _relationship_target(
    relationship: Dict[str, Any],
) -> Optional[str]:
    return _first_non_empty(
        relationship.get("target_entity_id"),
        relationship.get("to_entity_id"),
        relationship.get("object_id"),
        relationship.get("target"),
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
    """
    Correctly handle both:

        evidence_ids: ["EV1"]

    and:

        evidence_id: "EV1"
    """
    value = relationship.get("evidence_ids")

    if value is None:
        value = relationship.get("evidence_id")

    return _evidence_ids(value)


def _relationship_sentence(
    relationship: Dict[str, Any],
    entity_lookup: Dict[str, Dict[str, Any]],
) -> Optional[str]:
    source_id = _relationship_source(relationship)
    target_id = _relationship_target(relationship)

    source_name = _entity_display(
        source_id,
        entity_lookup,
    )

    target_name = _entity_display(
        target_id,
        entity_lookup,
    )

    if not source_name or not target_name:
        return None

    rel_type = _relationship_type(
        relationship
    )

    status = _relationship_status(
        relationship
    )

    templates = {
        "claims_role": (
            f"{source_name} is described as being in the role of "
            f"{target_name}."
        ),
        "claims_identity": (
            f"{source_name} is claiming the identity of "
            f"{target_name}."
        ),
        "associated_with": (
            f"{source_name} is associated with {target_name}."
        ),
        "represents": (
            f"{source_name} represents {target_name}."
        ),
        "works_for": (
            f"{source_name} is described as working for "
            f"{target_name}."
        ),
        "belongs_to": (
            f"{source_name} is linked to {target_name}."
        ),
        "sent": (
            f"{source_name} is shown as sending something to "
            f"{target_name}."
        ),
        "received": (
            f"{source_name} is shown as receiving something from "
            f"{target_name}."
        ),
        "sent_by": (
            f"{source_name} is shown as being sent by "
            f"{target_name}."
        ),
        "requested": (
            f"{source_name} is requesting something from "
            f"{target_name}."
        ),
        "requests": (
            f"{source_name} requests something from "
            f"{target_name}."
        ),
        "attached_to": (
            f"{source_name} is attached to {target_name}."
        ),
        "references": (
            f"{source_name} references {target_name}."
        ),
        "contains": (
            f"{source_name} contains information related to "
            f"{target_name}."
        ),
        "issued_by": (
            f"{source_name} is described as being issued by "
            f"{target_name}."
        ),
        "paid_to": (
            f"{source_name} is associated with payment to "
            f"{target_name}."
        ),
        "transferred_to": (
            f"{source_name} is associated with a transfer to "
            f"{target_name}."
        ),
        "uses": (
            f"{source_name} uses {target_name}."
        ),
        "mentions": (
            f"{source_name} mentions {target_name}."
        ),
        "related_to": (
            f"{source_name} is related to {target_name}."
        ),
    }

    sentence = templates.get(
        rel_type,
        (
            f"{source_name} has a "
            f"{rel_type.replace('_', ' ')} relationship "
            f"with {target_name}."
        ),
    )

    if status == "claimed":
        sentence = (
            sentence.rstrip(".")
            + " This is currently a claim, not a verified fact."
        )

    elif status == "inferred":
        sentence = (
            sentence.rstrip(".")
            + " This relationship was inferred from "
              "the available evidence."
        )

    elif status == "contradicted":
        sentence = (
            sentence.rstrip(".")
            + " The available evidence conflicts with "
              "this relationship."
        )

    return sentence


# ---------------------------------------------------------------------------
# Scenario helpers
# ---------------------------------------------------------------------------

def _scenario_data(
    scenario: Any,
) -> Dict[str, Any]:
    """Normalize nested or flat scenario output."""
    if not isinstance(scenario, dict):
        return {}

    nested = scenario.get("scenario")

    if isinstance(nested, dict):
        return nested

    return scenario


def _scenario_type(
    scenario: Dict[str, Any],
) -> str:
    return _lower(
        _first_non_empty(
            scenario.get("scenario_type"),
            "unknown",
        )
    )


def _scenario_title(
    scenario: Dict[str, Any],
) -> Optional[str]:
    return _first_non_empty(
        scenario.get("title"),
        scenario.get("scenario_title"),
    )


def _scenario_summary(
    scenario: Dict[str, Any],
) -> Optional[str]:
    return _first_non_empty(
        scenario.get("summary"),
        scenario.get("explanation"),
    )


def _scenario_evidence_ids(
    scenario: Dict[str, Any],
) -> List[str]:
    """
    Correctly normalize scenario evidence_ids.
    """
    value = scenario.get("evidence_ids")

    if value is None:
        value = scenario.get("supporting_evidence_ids")

    return _evidence_ids(value)


# ---------------------------------------------------------------------------
# User context
# ---------------------------------------------------------------------------

def _normalize_user_context(
    user_context: Any,
) -> List[str]:
    """
    Normalize user annotations.

    User context remains a claim/annotation and is never automatically
    treated as verified fact.
    """
    if user_context is None:
        return []

    if isinstance(user_context, str):
        return [
            user_context.strip()
        ] if user_context.strip() else []

    if isinstance(user_context, dict):
        result: List[str] = []

        for key in (
            "text",
            "annotation",
            "context",
            "claim",
            "value",
        ):
            text = _safe_text(
                user_context.get(key)
            )

            if text:
                result.append(text)

        return _unique(result)

    if isinstance(user_context, (list, tuple)):
        result: List[str] = []

        for item in user_context:
            if isinstance(item, dict):
                result.extend(
                    _normalize_user_context(item)
                )
            else:
                text = _safe_text(item)

                if text:
                    result.append(text)

        return _unique(result)

    text = _safe_text(user_context)

    return [text] if text else []


# ---------------------------------------------------------------------------
# Actor helpers
# ---------------------------------------------------------------------------

def _find_person_entities(
    entities: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    return [
        entity
        for entity in entities
        if _entity_type(entity) == "person"
    ]


def _extract_claimed_role_relationship(
    relationships: Sequence[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    for relationship in relationships:
        if _relationship_type(relationship) == "claims_role":
            return relationship

    return None


def _extract_identity_relationship(
    relationships: Sequence[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    for relationship in relationships:
        if _relationship_type(relationship) == "claims_identity":
            return relationship

    return None


def _resolve_primary_actor(
    entities: Sequence[Dict[str, Any]],
    relationships: Sequence[Dict[str, Any]],
    entity_lookup: Dict[str, Dict[str, Any]],
) -> Tuple[Optional[str], List[str]]:
    """
    Resolve the primary actor conservatively.

    Preference:
    1. claims_role source
    2. claims_identity source
    3. first person entity
    """
    trace_ids: List[str] = []

    role_relationship = _extract_claimed_role_relationship(
        relationships
    )

    if role_relationship:
        actor_id = _relationship_source(
            role_relationship
        )

        if actor_id:
            trace_ids.extend(
                _relationship_evidence_ids(
                    role_relationship
                )
            )

            actor_name = _entity_display(
                actor_id,
                entity_lookup,
            )

            if actor_name:
                return (
                    actor_name,
                    _unique(trace_ids),
                )

    identity_relationship = _extract_identity_relationship(
        relationships
    )

    if identity_relationship:
        actor_id = _relationship_source(
            identity_relationship
        )

        if actor_id:
            trace_ids.extend(
                _relationship_evidence_ids(
                    identity_relationship
                )
            )

            actor_name = _entity_display(
                actor_id,
                entity_lookup,
            )

            if actor_name:
                return (
                    actor_name,
                    _unique(trace_ids),
                )

    people = _find_person_entities(
        entities
    )

    if people:
        actor_name = _entity_name(
            people[0]
        )

        if actor_name:
            return actor_name, []

    return None, []


# ---------------------------------------------------------------------------
# Action helpers
# ---------------------------------------------------------------------------

def _contains_term(
    text: str,
    terms: Sequence[str],
) -> bool:
    text_lower = text.lower()

    return any(
        term in text_lower
        for term in terms
    )


def _collect_signal_text(
    evidence: Sequence[Dict[str, Any]],
    scenario: Dict[str, Any],
) -> str:
    parts: List[str] = []

    for value in (
        scenario.get("title"),
        scenario.get("summary"),
        scenario.get("explanation"),
    ):
        text = _safe_text(value)

        if text:
            parts.append(text)

    for item in evidence:
        for key in (
            "text",
            "content",
            "summary",
            "description",
            "message",
            "transcript",
            "ocr_text",
            "classification",
            "intent",
            "threat_type",
            "visual_classification",
        ):
            text = _safe_text(
                item.get(key)
            )

            if text:
                parts.append(text)

    parts.extend(
        _string_list(
            scenario.get("supporting_signals")
        )
    )

    return " ".join(parts)


def _infer_action(
    evidence: Sequence[Dict[str, Any]],
    scenario: Dict[str, Any],
    events: Sequence[Dict[str, Any]],
) -> Optional[str]:
    text = _collect_signal_text(
        evidence,
        scenario,
    )

    event_parts: List[str] = []

    for event in events:
        for key in (
            "event_type",
            "type",
            "action",
            "description",
            "summary",
        ):
            value = _safe_text(
                event.get(key)
            )

            if value:
                event_parts.append(value)

    text = f"{text} {' '.join(event_parts)}"

    if _contains_term(
        text,
        CREDENTIAL_TERMS,
    ):
        return (
            "provide or disclose a credential, "
            "verification code, or authentication secret"
        )

    if _contains_term(
        text,
        FINANCIAL_TERMS,
    ):
        return (
            "make or authorize a financial transaction"
        )

    if _contains_term(
        text,
        DOCUMENT_TERMS,
    ):
        return (
            "review, approve, or act on the supplied "
            "document or payment request"
        )

    return None


def _infer_pressure(
    evidence: Sequence[Dict[str, Any]],
    scenario: Dict[str, Any],
) -> bool:
    return _contains_term(
        _collect_signal_text(
            evidence,
            scenario,
        ),
        PRESSURE_TERMS,
    )


# ---------------------------------------------------------------------------
# Next-action reasoning
# ---------------------------------------------------------------------------

def _derive_likely_next_harmful_action(
    action: Optional[str],
    scenario: Dict[str, Any],
    evidence: Sequence[Dict[str, Any]],
) -> Optional[str]:
    """
    Describe a plausible next harmful action.

    The language intentionally uses "could" rather than presenting
    the future action as guaranteed.
    """
    scenario_type = _scenario_type(
        scenario
    )

    signal_text = _collect_signal_text(
        evidence,
        scenario,
    )

    if (
        "credential" in scenario_type
        or _contains_term(
            signal_text,
            CREDENTIAL_TERMS,
        )
    ):
        return (
            "The next harmful step could be disclosure of an OTP, "
            "password, PIN, or other authentication secret."
        )

    if (
        "payment" in scenario_type
        or "fraud" in scenario_type
        or _contains_term(
            signal_text,
            FINANCIAL_TERMS,
        )
    ):
        return (
            "The next harmful step could be sending money, "
            "approving a transaction, or changing payment details."
        )

    if "account_takeover" in scenario_type:
        return (
            "The next harmful step could be disclosure of credentials "
            "or completion of an account-verification step."
        )

    if "document_manipulation" in scenario_type:
        return (
            "The next harmful step could be acting on a manipulated "
            "or misleading document."
        )

    if "impersonation" in scenario_type:
        if action:
            return (
                f"The next harmful step could be to {action} "
                "based on the claimed identity."
            )

        return (
            "The next harmful step could be carrying out the request "
            "before the claimed identity is independently verified."
        )

    if action:
        return (
            f"The next harmful step could be to {action} "
            "before the claim is independently verified."
        )

    return None


# ---------------------------------------------------------------------------
# Stage creation
# ---------------------------------------------------------------------------

def _make_stage(
    *,
    stage_type: str,
    title: str,
    description: str,
    evidence_ids: Optional[Sequence[str]] = None,
    confidence: Optional[float] = None,
    status: str = "informational",
    source: str = "TrustPath",
) -> Dict[str, Any]:
    stage: Dict[str, Any] = {
        "stage_type": stage_type,
        "title": title,
        "description": description,
        "evidence_ids": _evidence_ids(
            evidence_ids
        ),
        "status": status,
        "source": source,
    }

    if confidence is not None:
        stage["confidence"] = round(
            max(
                0.0,
                min(
                    1.0,
                    float(confidence),
                ),
            ),
            4,
        )

    return stage


def _build_identity_stage(
    primary_actor: Optional[str],
    role_relationship: Optional[Dict[str, Any]],
    identity_relationship: Optional[Dict[str, Any]],
    entity_lookup: Dict[str, Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if not primary_actor:
        return None

    if role_relationship:
        target_id = _relationship_target(
            role_relationship
        )

        role_name = (
            _entity_display(
                target_id,
                entity_lookup,
            )
            if target_id
            else None
        )

        if role_name:
            return _make_stage(
                stage_type="identity",
                title="Identity or Role Claim",
                description=(
                    f"{primary_actor} is presented as {role_name}. "
                    "This relationship is treated as a claim unless "
                    "independently verified."
                ),
                evidence_ids=_relationship_evidence_ids(
                    role_relationship
                ),
                status="claimed",
            )

    if identity_relationship:
        target_id = _relationship_target(
            identity_relationship
        )

        target_name = (
            _entity_display(
                target_id,
                entity_lookup,
            )
            if target_id
            else None
        )

        if target_name:
            return _make_stage(
                stage_type="identity",
                title="Identity Claim",
                description=(
                    f"{primary_actor} is presented as being "
                    f"{target_name}. The identity claim is not "
                    "treated as verified by TrustPath."
                ),
                evidence_ids=_relationship_evidence_ids(
                    identity_relationship
                ),
                status="claimed",
            )

    return _make_stage(
        stage_type="identity",
        title="Communicating Party",
        description=(
            f"The available evidence identifies {primary_actor} "
            "as a relevant communicating party."
        ),
        status="context",
    )


def _build_relationship_stage(
    relationships: Sequence[Dict[str, Any]],
    entity_lookup: Dict[str, Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    sentences: List[str] = []
    evidence_ids: List[str] = []

    for relationship in relationships:
        sentence = _relationship_sentence(
            relationship,
            entity_lookup,
        )

        if sentence:
            sentences.append(sentence)

            evidence_ids.extend(
                _relationship_evidence_ids(
                    relationship
                )
            )

        if len(sentences) >= 3:
            break

    if not sentences:
        return None

    return _make_stage(
        stage_type="relationship",
        title="Relationship Evidence",
        description=" ".join(sentences),
        evidence_ids=evidence_ids,
        status="inferred",
    )


def _build_request_stage(
    action: Optional[str],
    evidence: Sequence[Dict[str, Any]],
    scenario: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    if not action:
        return None

    evidence_ids: List[str] = []

    for item in evidence:
        detected_data = item.get("detected_data")
        if not isinstance(detected_data, dict):
            detected_data = {}

        # Build one evidence signal from both the normalized evidence
        # object and its preserved detector payload. The generic adapter
        # description (for example, "Existing Text analysis supplied...")
        # must not hide the actual request text stored in detected_data.
        signal_parts: List[str] = []

        for value in (
            item.get("text"),
            item.get("content"),
            item.get("summary"),
            item.get("message"),
            item.get("transcript"),
            item.get("ocr_text"),
            item.get("classification"),
            item.get("intent"),
            item.get("description"),
            detected_data.get("text"),
            detected_data.get("content"),
            detected_data.get("summary"),
            detected_data.get("message"),
            detected_data.get("transcript"),
            detected_data.get("ocr_text"),
            detected_data.get("classification"),
            detected_data.get("intent"),
            detected_data.get("event_title"),
            detected_data.get("event_description"),
            detected_data.get("action"),
        ):
            text = _safe_text(value)
            if text:
                signal_parts.append(text)

        # Also inspect any explicitly supplied event object.
        for event_value in (
            detected_data.get("event"),
            detected_data.get("request"),
        ):
            if isinstance(event_value, dict):
                for key in (
                    "title",
                    "description",
                    "summary",
                    "action",
                    "type",
                    "event_type",
                ):
                    text = _safe_text(event_value.get(key))
                    if text:
                        signal_parts.append(text)

        signal_text = _lower(" ".join(_unique(signal_parts)))

        if not signal_text:
            continue

        if (
            _contains_term(signal_text, FINANCIAL_TERMS)
            or _contains_term(signal_text, CREDENTIAL_TERMS)
            or _contains_term(signal_text, DOCUMENT_TERMS)
        ):
            evidence_id = _evidence_id(item)

            if evidence_id:
                evidence_ids.append(evidence_id)

    evidence_ids = _unique(evidence_ids)

    pressure_text = ""

    if _infer_pressure(
        evidence,
        scenario,
    ):
        pressure_text = (
            " The communication also contains signs of urgency "
            "or pressure."
        )

    return _make_stage(
        stage_type="request",
        title="Requested Action",
        description=(
            f"The communication appears to ask the recipient to "
            f"{action}.{pressure_text}"
        ),
        evidence_ids=evidence_ids,
        status="observed",
    )

def _event_description(
    event: Dict[str, Any],
) -> Optional[str]:
    return _first_non_empty(
        event.get("description"),
        event.get("summary"),
        event.get("action"),
        event.get("event_description"),
    )


def _event_type(
    event: Dict[str, Any],
) -> Optional[str]:
    return _first_non_empty(
        event.get("event_type"),
        event.get("type"),
        event.get("name"),
    )


def _event_evidence_ids(
    event: Dict[str, Any],
) -> List[str]:
    """
    Correct handling for event evidence_ids.
    """
    value = event.get("evidence_ids")

    if value is None:
        value = event.get("evidence_id")

    return _evidence_ids(value)


def _build_event_stages(
    events: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    stages: List[Dict[str, Any]] = []

    for event in events:
        description = _event_description(event)

        if not description:
            event_type = _event_type(event)

            if event_type:
                description = (
                    event_type
                    .replace("_", " ")
                    .capitalize()
                )

        if not description:
            continue

        event_type = (
            _event_type(event)
            or "event"
        )

        stages.append(
            _make_stage(
                stage_type="action",
                title=(
                    event_type
                    .replace("_", " ")
                    .title()
                ),
                description=description,
                evidence_ids=_event_evidence_ids(
                    event
                ),
                status="observed",
            )
        )

    return stages


# ---------------------------------------------------------------------------
# Contradiction helpers
# ---------------------------------------------------------------------------

def _contradiction_description(
    contradiction: Dict[str, Any],
) -> Optional[str]:
    return _first_non_empty(
        contradiction.get("description"),
        contradiction.get("summary"),
        contradiction.get("explanation"),
        contradiction.get("reason"),
        contradiction.get("conflict"),
    )


def _contradiction_evidence_ids(
    contradiction: Dict[str, Any],
) -> List[str]:
    """
    Correct handling for contradiction evidence_ids.
    """
    value = contradiction.get("evidence_ids")

    if value is None:
        value = contradiction.get("evidence_id")

    if value is None:
        value = contradiction.get(
            "supporting_evidence_ids"
        )

    return _evidence_ids(value)


def _build_contradiction_stages(
    contradictions: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    stages: List[Dict[str, Any]] = []

    for contradiction in contradictions:
        description = _contradiction_description(
            contradiction
        )

        if not description:
            continue

        stages.append(
            _make_stage(
                stage_type="contradiction",
                title="Evidence Conflict",
                description=description,
                evidence_ids=_contradiction_evidence_ids(
                    contradiction
                ),
                status="contradiction",
            )
        )

    return stages


# ---------------------------------------------------------------------------
# Risk stage
# ---------------------------------------------------------------------------

def _build_assessment_phrase(
    scenario: Dict[str, Any],
) -> str:
    scenario_type = _scenario_type(
        scenario
    )

    title = _scenario_title(
        scenario
    )

    if title:
        return title

    mappings = {
        "possible_impersonation": "possible impersonation",
        "possible_payment_diversion": "possible payment diversion",
        "possible_credential_request": "possible credential request",
        "possible_social_engineering": "possible social engineering",
        "possible_document_manipulation": "possible document manipulation",
        "possible_account_takeover": "possible account takeover",
        "possible_fraud": "possible fraud",
        "evidence_inconsistency": "an evidence inconsistency",
        "communication_event": "a communication event",
    }

    return mappings.get(
        scenario_type,
        "an unclear communication scenario",
    )


def _build_risk_stage(
    scenario: Dict[str, Any],
    risk_level: str,
) -> Dict[str, Any]:
    summary = _scenario_summary(
        scenario
    )

    scenario_phrase = _build_assessment_phrase(
        scenario
    )

    parts = [
        f"TrustPath reconstructs this as "
        f"{scenario_phrase.lower()}."
    ]

    if summary:
        parts.append(summary)

    if risk_level != "UNKNOWN":
        parts.append(
            f"Current scenario risk is assessed as "
            f"{risk_level}."
        )

    confidence = scenario.get(
        "confidence"
    )

    try:
        confidence_value = (
            float(confidence)
            if confidence is not None
            else None
        )
    except (TypeError, ValueError):
        confidence_value = None

    return _make_stage(
        stage_type="risk",
        title="Scenario Assessment",
        description=" ".join(parts),
        evidence_ids=_scenario_evidence_ids(
            scenario
        ),
        confidence=confidence_value,
        status="assessment",
    )


# ---------------------------------------------------------------------------
# Next action / protection stages
# ---------------------------------------------------------------------------

def _build_next_action_stage(
    next_action: Optional[str],
    evidence_ids: Sequence[str],
) -> Optional[Dict[str, Any]]:
    if not next_action:
        return None

    return _make_stage(
        stage_type="next_action",
        title="Possible Next Harmful Step",
        description=next_action,
        evidence_ids=evidence_ids,
        status="possible",
    )


def _build_protection_stage(
    scenario: Dict[str, Any],
) -> Dict[str, Any]:
    scenario_type = _scenario_type(
        scenario
    )

    if (
        "payment" in scenario_type
        or "fraud" in scenario_type
        or "impersonation" in scenario_type
    ):
        protection = (
            "Pause before transferring money or changing payment "
            "details. Verify the claimed sender through an "
            "independent channel."
        )

    elif (
        "credential" in scenario_type
        or "account_takeover" in scenario_type
    ):
        protection = (
            "Do not disclose passwords, OTPs, PINs, or verification "
            "codes. Verify the request through a trusted independent "
            "channel."
        )

    elif "document_manipulation" in scenario_type:
        protection = (
            "Do not approve the document yet. Verify the document "
            "source, payment details, and sender independently."
        )

    else:
        protection = (
            "Pause the requested action and independently verify "
            "the identity, relationship, and request before "
            "proceeding."
        )

    return _make_stage(
        stage_type="protection",
        title="Recommended Protection",
        description=protection,
        evidence_ids=_scenario_evidence_ids(
            scenario
        ),
        status="recommendation",
    )


# ---------------------------------------------------------------------------
# Evidence blocks
# ---------------------------------------------------------------------------

def _build_evidence_blocks(
    evidence: Sequence[Dict[str, Any]],
    contradictions: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []

    contradiction_ids = set()

    for contradiction in contradictions:
        contradiction_ids.update(
            _contradiction_evidence_ids(
                contradiction
            )
        )

    for item in evidence:
        evidence_id = _evidence_id(item)

        if not evidence_id:
            continue

        block = {
            "evidence_id": evidence_id,
            "source": _evidence_source(item),
            "source_label": _source_label(
                _evidence_source(item)
            ),
            "classification": _evidence_classification(
                item
            ),
            "confidence": _evidence_confidence(
                item
            ),
            "excerpt": _evidence_excerpt(
                item
            ),
            "role": (
                "contradicting"
                if evidence_id in contradiction_ids
                else "supporting"
            ),
        }

        blocks.append(block)

    return blocks


def _build_story_blocks(
    stages: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []

    for position, stage in enumerate(
        stages,
        start=1,
    ):
        description = _safe_text(
            stage.get("description")
        )

        if not description:
            continue

        blocks.append(
            {
                "sequence": position,
                "stage_type": stage.get(
                    "stage_type"
                ),
                "title": stage.get(
                    "title"
                ),
                "text": description,
                "evidence_ids": _evidence_ids(
                    stage.get(
                        "evidence_ids"
                    )
                ),
                "status": stage.get(
                    "status"
                ),
            }
        )

    return blocks


# ---------------------------------------------------------------------------
# Top-level story text
# ---------------------------------------------------------------------------

def _build_headline(
    primary_actor: Optional[str],
    action: Optional[str],
    scenario: Dict[str, Any],
) -> str:
    scenario_type = _scenario_type(
        scenario
    )

    if primary_actor and "impersonation" in scenario_type:
        if action:
            return (
                f"Possible impersonation involving "
                f"{primary_actor} and a request to {action}"
            )

        return (
            f"Possible impersonation involving {primary_actor}"
        )

    if primary_actor and action:
        return (
            f"{primary_actor} is associated with a request to "
            f"{action}"
        )

    scenario_title = _scenario_title(
        scenario
    )

    if scenario_title:
        return scenario_title

    return (
        "TrustPath reconstructed a potentially risky interaction"
    )


def _build_summary(
    primary_actor: Optional[str],
    action: Optional[str],
    scenario: Dict[str, Any],
    risk_level: str,
    contradictions: Sequence[Dict[str, Any]],
) -> str:
    scenario_phrase = _build_assessment_phrase(
        scenario
    ).lower()

    parts: List[str] = []

    if primary_actor:
        parts.append(
            f"The evidence centers on {primary_actor}"
        )
    else:
        parts.append(
            "The available evidence centers on "
            "an unidentified communicating party"
        )

    if action:
        parts.append(
            f"with a request to {action}"
        )

    parts.append(
        f"within a {scenario_phrase} scenario"
    )

    sentence = " ".join(parts).rstrip(".") + "."

    if contradictions:
        count = len(contradictions)

        sentence += (
            f" {count} evidence conflict"
            + (
                "s"
                if count != 1
                else ""
            )
            + " were identified."
        )

    if risk_level != "UNKNOWN":
        sentence += (
            f" The current assessed risk is "
            f"{risk_level.lower()}."
        )

    return sentence


def _build_current_manipulation(
    scenario: Dict[str, Any],
    evidence: Sequence[Dict[str, Any]],
) -> str:
    scenario_type = _scenario_type(
        scenario
    )

    if "impersonation" in scenario_type:
        return (
            "The interaction appears to rely on a claimed identity "
            "or relationship to make the request more believable."
        )

    if "payment_diversion" in scenario_type:
        return (
            "The interaction appears to steer the recipient toward "
            "a financial action or changed payment destination."
        )

    if "credential" in scenario_type:
        return (
            "The interaction appears to create a reason for the "
            "recipient to disclose authentication information."
        )

    if "social_engineering" in scenario_type:
        return (
            "The interaction appears designed to influence the "
            "recipient through urgency, authority, trust, or "
            "other social pressure."
        )

    if "document_manipulation" in scenario_type:
        return (
            "The evidence suggests the recipient may be asked to "
            "trust or act on a potentially manipulated document."
        )

    if _infer_pressure(
        evidence,
        scenario,
    ):
        return (
            "The interaction contains pressure signals that may "
            "encourage the recipient to act before independently "
            "verifying the claim."
        )

    return (
        "TrustPath does not have enough evidence to identify a "
        "specific manipulation technique with confidence."
    )


def _build_victim_decision(
    action: Optional[str],
) -> str:
    if action:
        return (
            f"The recipient is being asked to decide whether to "
            f"{action}."
        )

    return (
        "The recipient is being placed in a position where they "
        "may need to decide whether to act on the communication."
    )


# ---------------------------------------------------------------------------
# Traceability
# ---------------------------------------------------------------------------

def _validate_story_traceability(
    stages: Sequence[Dict[str, Any]],
    evidence_ids: Sequence[str],
) -> Dict[str, Any]:
    known_ids = set(
        _evidence_ids(evidence_ids)
    )

    referenced_ids: List[str] = []

    for stage in stages:
        referenced_ids.extend(
            _evidence_ids(
                stage.get("evidence_ids")
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
# Public validation
# ---------------------------------------------------------------------------

def validate_story(
    story: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Validate story structure and evidence traceability.
    """
    if not isinstance(story, dict):
        return {
            "valid": False,
            "errors": [
                "Story must be a dictionary."
            ],
        }

    required_fields = (
        "headline",
        "summary",
        "current_manipulation",
        "victim_decision",
        "likely_next_harmful_action",
        "stages",
        "evidence_blocks",
        "traceability",
    )

    errors: List[str] = []

    for field in required_fields:
        if field not in story:
            errors.append(
                f"Missing required story field: {field}"
            )

    if not isinstance(
        story.get("stages"),
        list,
    ):
        errors.append(
            "Story stages must be a list."
        )

    if not isinstance(
        story.get("evidence_blocks"),
        list,
    ):
        errors.append(
            "Story evidence_blocks must be a list."
        )

    traceability = story.get(
        "traceability"
    )

    if not isinstance(
        traceability,
        dict,
    ):
        errors.append(
            "Story traceability must be a dictionary."
        )
    elif traceability.get("valid") is False:
        errors.append(
            "Story contains invalid evidence references."
        )

    return {
        "valid": len(errors) == 0,
        "errors": errors,
    }


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_trustpath_story(
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
    contradictions: Optional[
        Sequence[Dict[str, Any]]
    ] = None,
    scenario: Optional[
        Dict[str, Any]
    ] = None,
    risk_level: Optional[str] = None,
    user_context: Any = None,
) -> Dict[str, Any]:
    """
    Build the complete deterministic TrustPath story.
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

    normalized_contradictions = _normalize_contradictions(
        contradictions
    )

    scenario_data = _scenario_data(
        scenario or {}
    )

    entity_lookup = _build_entity_lookup(
        normalized_entities
    )

    normalized_risk = _risk_level(
        risk_level
        or scenario_data.get(
            "risk_level"
        )
        or "UNKNOWN"
    )

    user_context_values = _normalize_user_context(
        user_context
    )

    # ---------------------------------------------------------------
    # Primary actor
    # ---------------------------------------------------------------

    role_relationship = _extract_claimed_role_relationship(
        normalized_relationships
    )

    identity_relationship = _extract_identity_relationship(
        normalized_relationships
    )

    primary_actor, actor_trace = _resolve_primary_actor(
        normalized_entities,
        normalized_relationships,
        entity_lookup,
    )

    # ---------------------------------------------------------------
    # Requested action
    # ---------------------------------------------------------------

    action = _infer_action(
        normalized_evidence,
        scenario_data,
        normalized_events,
    )

    next_action = _derive_likely_next_harmful_action(
        action,
        scenario_data,
        normalized_evidence,
    )

    # ---------------------------------------------------------------
    # Dynamic stages
    # ---------------------------------------------------------------

    stages: List[Dict[str, Any]] = []

    identity_stage = _build_identity_stage(
        primary_actor,
        role_relationship,
        identity_relationship,
        entity_lookup,
    )

    if identity_stage:
        stages.append(identity_stage)

    relationship_stage = _build_relationship_stage(
        normalized_relationships,
        entity_lookup,
    )

    if relationship_stage:
        stages.append(
            relationship_stage
        )

    request_stage = _build_request_stage(
        action,
        normalized_evidence,
        scenario_data,
    )

    if request_stage:
        stages.append(
            request_stage
        )

    stages.extend(
        _build_event_stages(
            normalized_events
        )
    )

    stages.extend(
        _build_contradiction_stages(
            normalized_contradictions
        )
    )

    stages.append(
        _build_risk_stage(
            scenario_data,
            normalized_risk,
        )
    )

    next_action_evidence_ids = _unique(
        _scenario_evidence_ids(
            scenario_data
        )
        + _evidence_ids(
            actor_trace
        )
    )

    next_action_stage = _build_next_action_stage(
        next_action,
        next_action_evidence_ids,
    )

    if next_action_stage:
        stages.append(
            next_action_stage
        )

    stages.append(
        _build_protection_stage(
            scenario_data
        )
    )

    # ---------------------------------------------------------------
    # Evidence-backed blocks
    # ---------------------------------------------------------------

    evidence_blocks = _build_evidence_blocks(
        normalized_evidence,
        normalized_contradictions,
    )

    story_blocks = _build_story_blocks(
        stages
    )

    # ---------------------------------------------------------------
    # Top-level story
    # ---------------------------------------------------------------

    headline = _build_headline(
        primary_actor,
        action,
        scenario_data,
    )

    summary = _build_summary(
        primary_actor,
        action,
        scenario_data,
        normalized_risk,
        normalized_contradictions,
    )

    current_manipulation = _build_current_manipulation(
        scenario_data,
        normalized_evidence,
    )

    victim_decision = _build_victim_decision(
        action
    )

    # ---------------------------------------------------------------
    # Traceability
    # ---------------------------------------------------------------

    known_evidence_ids = [
        evidence_id
        for evidence_id in (
            _evidence_id(item)
            for item in normalized_evidence
        )
        if evidence_id
    ]

    traceability = _validate_story_traceability(
        stages,
        known_evidence_ids,
    )

    traceability.update(
        {
            "story_block_count": len(
                story_blocks
            ),
            "stage_count": len(
                stages
            ),
            "evidence_block_count": len(
                evidence_blocks
            ),
            "user_context_count": len(
                user_context_values
            ),
            "user_context_is_claim_only": True,
        }
    )

    # ---------------------------------------------------------------
    # Final story object
    # ---------------------------------------------------------------

    story: Dict[str, Any] = {
        "headline": headline,
        "summary": summary,
        "current_manipulation": current_manipulation,
        "victim_decision": victim_decision,
        "likely_next_harmful_action": next_action,
        "risk_level": normalized_risk,
        "scenario_type": _scenario_type(
            scenario_data
        ),
        "scenario_title": _scenario_title(
            scenario_data
        ),
        "stages": stages,
        "story_blocks": story_blocks,
        "evidence_blocks": evidence_blocks,
        "user_context": [
            {
                "text": text,
                "status": "user_claim",
            }
            for text in user_context_values
        ],
        "traceability": traceability,
        "metadata": {
            "builder": "trustpath.story",
            "deterministic": True,
            "llm_used": False,
            "prediction_language": "possible",
            "identity_claims_verified": False,
            "relationship_claims_verified": False,
        },
    }

    return story


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def build_story_from_reasoning(
    reasoning: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build a story directly from an orchestrator-style reasoning object.
    """
    if not isinstance(reasoning, dict):
        reasoning = {}

    return build_trustpath_story(
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
        contradictions=reasoning.get(
            "contradictions"
        ),
        scenario=reasoning.get(
            "scenario"
        ),
        risk_level=reasoning.get(
            "risk_level"
        ),
        user_context=reasoning.get(
            "user_context"
        ),
    )


__all__ = [
    "build_trustpath_story",
    "build_story_from_reasoning",
    "validate_story",
]