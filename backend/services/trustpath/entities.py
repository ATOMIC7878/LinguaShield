"""
LinguaShield - TrustPath Entity Registry
-----------------------------------------

Purpose:
    Build a canonical registry of entities found in TrustPath evidence.

Architecture:

    Existing analysis / user evidence
                    ↓
                adapter.py
                    ↓
              normalized evidence
                    ↓
                entities.py
                    ↓
            canonical entities
                    ↓
          relationships.py

Important:
    - This module does NOT run Text, Visual, Acoustic, or Biometric models.
    - It only reads normalized TrustPath evidence.
    - It does NOT treat user-provided context as verified truth.
    - Duplicate entity mentions are merged where identity can be matched
      conservatively.
    - Every entity keeps traceability back to its evidence.

Expected project location:
    LinguaShield/backend/services/trustpath/entities.py
"""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Optional, Tuple
from uuid import uuid4


# ---------------------------------------------------------------------------
# ENTITY TYPES
# ---------------------------------------------------------------------------

ENTITY_TYPES = {
    "person",
    "organization",
    "brand",
    "phone",
    "email",
    "account",
    "transaction",
    "document",
    "invoice",
    "location",
    "product",
    "url",
    "unknown",
}


# ---------------------------------------------------------------------------
# STATUS VALUES
# ---------------------------------------------------------------------------

SOURCE_STATUSES = {
    "detected",
    "user_provided",
    "verified",
    "unverified",
    "contradicted",
    "unknown",
}


# ---------------------------------------------------------------------------
# INTERNAL HELPERS
# ---------------------------------------------------------------------------

def _new_id(prefix: str = "entity") -> str:
    """Create a readable TrustPath entity ID."""
    return f"{prefix}_{uuid4().hex[:12]}"


def _clean_string(value: Any) -> Optional[str]:
    """Return a stripped non-empty string or None."""
    if value is None:
        return None

    if not isinstance(value, str):
        value = str(value)

    value = value.strip()

    return value or None


def _normalize_name(value: Any) -> Optional[str]:
    """
    Normalize a name for conservative duplicate matching.

    This is NOT a fuzzy identity matcher.
    It only removes trivial formatting differences.
    """

    value = _clean_string(value)

    if value is None:
        return None

    value = value.casefold()

    # Collapse repeated whitespace.
    value = re.sub(r"\s+", " ", value)

    # Remove harmless surrounding punctuation.
    value = value.strip(" \t\r\n.,;:|")

    return value or None


def _normalize_identifier(
    value: Any,
    entity_type: str,
) -> Optional[str]:
    """
    Normalize identifiers used for deterministic matching.

    Emails and URLs are lower-cased.

    Phone numbers retain digits only.

    Other entity types use conservative name normalization.
    """

    value = _clean_string(value)

    if value is None:
        return None

    if entity_type == "email":
        return value.casefold()

    if entity_type == "url":
        return value.casefold().rstrip("/")

    if entity_type == "phone":
        digits = re.sub(r"\D", "", value)
        return digits or None

    return _normalize_name(value)


def _valid_entity_type(value: Any) -> str:
    """
    Convert an arbitrary entity type into a supported type.
    """

    if not isinstance(value, str):
        return "unknown"

    candidate = value.strip().casefold()

    aliases = {
        "person_name": "person",
        "human": "person",
        "individual": "person",
        "company": "organization",
        "org": "organization",
        "company_name": "organization",
        "business": "organization",
        "logo": "brand",
        "telephone": "phone",
        "phone_number": "phone",
        "email_address": "email",
        "bank_account": "account",
        "account_number": "account",
        "txn": "transaction",
        "transaction_id": "transaction",
        "doc": "document",
        "invoice_number": "invoice",
        "web_url": "url",
        "website": "url",
    }

    candidate = aliases.get(candidate, candidate)

    if candidate not in ENTITY_TYPES:
        return "unknown"

    return candidate


def _safe_confidence(value: Any) -> Optional[float]:
    """
    Normalize a confidence value to 0..1.

    Accepts:
        0..1
        0..100

    Invalid values become None.
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


def _safe_status(value: Any) -> str:
    """Return a supported evidence status."""
    if isinstance(value, str):
        candidate = value.strip().casefold()

        if candidate in SOURCE_STATUSES:
            return candidate

    return "unknown"


def _append_unique(
    target: List[str],
    values: Iterable[str],
) -> None:
    """Append values while preserving order and removing duplicates."""

    for value in values:
        if not value:
            continue

        if value not in target:
            target.append(value)


# ---------------------------------------------------------------------------
# ENTITY RECORD
# ---------------------------------------------------------------------------

class EntityRecord:
    """
    Internal mutable representation of a TrustPath entity.

    The final output is a plain dictionary so it can be consumed by the
    existing TrustPath Pydantic models without creating a dependency cycle.
    """

    def __init__(
        self,
        *,
        entity_id: str,
        entity_type: str,
        name: Optional[str],
        evidence_id: str,
        source: str,
        status: str,
        confidence: Optional[float],
    ) -> None:
        self.entity_id = entity_id
        self.entity_type = entity_type
        self.name = name

        self.source_evidence_ids: List[str] = []

        self.sources: List[str] = []

        self.attributes: Dict[str, Any] = {}

        self.confidence = confidence

        self.status = status

        self.user_claims: List[str] = []

        self.detected_values: List[str] = []

        self.add_evidence(
            evidence_id=evidence_id,
            source=source,
            status=status,
            confidence=confidence,
        )

    def add_evidence(
        self,
        *,
        evidence_id: Optional[str],
        source: Optional[str],
        status: Optional[str],
        confidence: Optional[float],
    ) -> None:
        """Attach another evidence source to this entity."""

        if evidence_id:
            _append_unique(
                self.source_evidence_ids,
                [evidence_id],
            )

        if source:
            _append_unique(
                self.sources,
                [source],
            )

        if status:
            normalized_status = _safe_status(status)

            # Prefer stronger states without treating them as absolute truth.
            if normalized_status == "verified":
                self.status = "verified"
            elif (
                self.status not in {"verified", "contradicted"}
                and normalized_status == "contradicted"
            ):
                self.status = "contradicted"
            elif (
                self.status == "unknown"
                and normalized_status != "unknown"
            ):
                self.status = normalized_status

        if confidence is not None:
            if self.confidence is None:
                self.confidence = confidence
            else:
                self.confidence = max(
                    self.confidence,
                    confidence,
                )

    def add_attribute(
        self,
        key: str,
        value: Any,
    ) -> None:
        """Store a non-empty entity attribute."""

        if value is None:
            return

        if isinstance(value, str):
            value = value.strip()

            if not value:
                return

        self.attributes[key] = deepcopy(value)

    def add_user_claim(
        self,
        claim: str,
    ) -> None:
        """Record context supplied by a user as a claim, not a verified fact."""

        clean = _clean_string(claim)

        if clean:
            _append_unique(
                self.user_claims,
                [clean],
            )

    def add_detected_value(
        self,
        value: str,
    ) -> None:
        """Record an observed representation of the entity."""

        clean = _clean_string(value)

        if clean:
            _append_unique(
                self.detected_values,
                [clean],
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert the entity into the canonical TrustPath representation."""

        attributes = deepcopy(self.attributes)

        if self.sources:
            attributes["sources"] = list(self.sources)

        if self.user_claims:
            attributes["user_claims"] = list(self.user_claims)

        if self.detected_values:
            attributes["detected_values"] = list(self.detected_values)

        attributes["evidence_status"] = self.status

        return {
            "entity_id": self.entity_id,
            "entity_type": self.entity_type,
            "name": self.name,
            "source_evidence_ids": list(self.source_evidence_ids),
            "attributes": attributes,
            "confidence": self.confidence,
        }


# ---------------------------------------------------------------------------
# CANDIDATE EXTRACTION
# ---------------------------------------------------------------------------

def _candidate_from_field(
    *,
    data: Dict[str, Any],
    field_name: str,
    entity_type: str,
) -> Optional[Tuple[str, str]]:
    """
    Extract a simple entity candidate from a known field.

    Returns:
        (normalized_value, display_value)
    """

    if field_name not in data:
        return None

    value = _clean_string(data.get(field_name))

    if value is None:
        return None

    normalized = _normalize_identifier(
        value,
        entity_type,
    )

    if normalized is None:
        return None

    return normalized, value


def _iter_known_fields(
    data: Dict[str, Any],
) -> Iterable[Tuple[str, str, Any]]:
    """
    Yield explicitly supported entity fields.

    The function is intentionally explicit to prevent arbitrary dictionary
    values from becoming entities.
    """

    field_map = {
        "person_name": "person",
        "sender_name": "person",
        "recipient_name": "person",
        "caller_name": "person",
        "speaker_name": "person",

        "brand_name": "brand",
        "brand": "brand",

        "company_name": "organization",
        "organization_name": "organization",
        "company": "organization",

        "phone": "phone",
        "phone_number": "phone",
        "telephone": "phone",

        "email": "email",
        "email_address": "email",

        "account": "account",
        "account_number": "account",
        "bank_account": "account",

        "transaction_id": "transaction",
        "transaction": "transaction",

        "document_id": "document",
        "document_name": "document",

        "invoice_number": "invoice",
        "invoice_id": "invoice",

        "location": "location",
        "location_name": "location",

        "product_name": "product",
        "product": "product",

        "url": "url",
        "website": "url",
        "web_url": "url",
    }

    for field_name, entity_type in field_map.items():
        if field_name in data:
            yield (
                field_name,
                entity_type,
                data.get(field_name),
            )


# ---------------------------------------------------------------------------
# USER CONTEXT PARSING
# ---------------------------------------------------------------------------

def _extract_user_claim_entity(
    *,
    context: Optional[str],
) -> Optional[Tuple[str, str, str]]:
    """
    Handle very simple explicit user context.

    Example:
        "Rahul, my manager"

    Returns:
        (name, role, original_context)

    This does NOT claim that Rahul is actually the manager.
    """

    context = _clean_string(context)

    if not context:
        return None

    patterns = [
        r"^(?P<name>[^,]+),\s*my\s+(?P<role>.+)$",
        r"^(?P<name>[^,]+),\s*(?P<role>.+)$",
    ]

    for pattern in patterns:
        match = re.match(
            pattern,
            context,
            flags=re.IGNORECASE,
        )

        if match:
            name = _clean_string(match.group("name"))
            role = _clean_string(match.group("role"))

            if name and role:
                return name, role, context

    return None


def apply_user_context(
    registry: Dict[Tuple[str, str], EntityRecord],
    *,
    context: Optional[str],
    evidence_id: Optional[str] = None,
) -> None:
    """
    Store simple user-provided identity context as an entity claim.

    The role is saved as:
        claimed_role

    rather than as a verified relationship.
    """

    extracted = _extract_user_claim_entity(
        context=context,
    )

    if extracted is None:
        return

    name, role, original_context = extracted

    normalized_name = _normalize_identifier(
        name,
        "person",
    )

    if normalized_name is None:
        return

    key = ("person", normalized_name)

    entity = registry.get(key)

    if entity is None:
        entity = EntityRecord(
            entity_id=_new_id(),
            entity_type="person",
            name=name,
            evidence_id=evidence_id or "",
            source="user",
            status="user_provided",
            confidence=None,
        )

        registry[key] = entity

    entity.add_attribute(
        "claimed_role",
        role,
    )

    entity.add_user_claim(
        original_context,
    )


# ---------------------------------------------------------------------------
# EVIDENCE TO ENTITY CONVERSION
# ---------------------------------------------------------------------------

def extract_entities_from_evidence(
    evidence: Iterable[Any],
) -> List[Dict[str, Any]]:
    """
    Build canonical entities from normalized TrustPath evidence.

    Accepted evidence:
        - Pydantic EvidenceItem objects
        - dictionaries containing the same fields

    The function is deliberately conservative:
        - only known fields become entities
        - arbitrary text is not guessed into identities
        - user context is kept as a claim
    """

    registry: Dict[Tuple[str, str], EntityRecord] = {}

    for item in evidence:
        item_dict = _as_dict(item)

        if not item_dict:
            continue

        evidence_id = _clean_string(
            item_dict.get("evidence_id")
        ) or ""

        source = _clean_string(
            item_dict.get("source")
        ) or "system"

        status = _safe_status(
            item_dict.get("status")
        )

        evidence_confidence = _safe_confidence(
            item_dict.get("confidence")
        )

        user_context = _clean_string(
            item_dict.get("user_context")
        )

        detected_data = item_dict.get(
            "detected_data",
            {},
        )

        if not isinstance(detected_data, dict):
            detected_data = {}

        # ---------------------------------------------------------------
        # Explicitly structured entity fields
        # ---------------------------------------------------------------

        for field_name, entity_type, raw_value in _iter_known_fields(
            detected_data
        ):
            normalized = _normalize_identifier(
                raw_value,
                entity_type,
            )

            display_value = _clean_string(
                raw_value
            )

            if normalized is None or display_value is None:
                continue

            key = (
                entity_type,
                normalized,
            )

            entity = registry.get(key)

            if entity is None:
                entity = EntityRecord(
                    entity_id=_new_id(),
                    entity_type=entity_type,
                    name=display_value,
                    evidence_id=evidence_id,
                    source=source,
                    status=status,
                    confidence=evidence_confidence,
                )

                registry[key] = entity

            else:
                entity.add_evidence(
                    evidence_id=evidence_id,
                    source=source,
                    status=status,
                    confidence=evidence_confidence,
                )

            entity.add_detected_value(
                display_value
            )

            entity.add_attribute(
                "detected_field",
                field_name,
            )

            # Preserve useful original information where present.
            if field_name in {
                "transaction_id",
                "invoice_number",
                "document_id",
            }:
                entity.add_attribute(
                    field_name,
                    display_value,
                )

        # ---------------------------------------------------------------
        # Common structured nested entity lists
        # ---------------------------------------------------------------

        _extract_nested_entities(
            registry=registry,
            detected_data=detected_data,
            evidence_id=evidence_id,
            source=source,
            status=status,
            confidence=evidence_confidence,
        )

        # ---------------------------------------------------------------
        # User-supplied context
        # ---------------------------------------------------------------

        apply_user_context(
            registry,
            context=user_context,
            evidence_id=evidence_id,
        )

        # ---------------------------------------------------------------
        # Preserve explicit user claims inside detected_data
        # ---------------------------------------------------------------

        explicit_claim = _clean_string(
            detected_data.get("user_claim")
        )

        if explicit_claim:
            # Attach to the most obvious person entity from this evidence.
            person_entities = [
                entity
                for entity in registry.values()
                if (
                    entity.entity_type == "person"
                    and evidence_id in entity.source_evidence_ids
                )
            ]

            if person_entities:
                person_entities[0].add_user_claim(
                    explicit_claim
                )

    return [
        entity.to_dict()
        for entity in registry.values()
    ]


# ---------------------------------------------------------------------------
# NESTED ENTITY EXTRACTION
# ---------------------------------------------------------------------------

def _extract_nested_entities(
    *,
    registry: Dict[Tuple[str, str], EntityRecord],
    detected_data: Dict[str, Any],
    evidence_id: str,
    source: str,
    status: str,
    confidence: Optional[float],
) -> None:
    """
    Extract entities from explicitly structured lists.

    Supported examples:

        {
            "entities": [
                {
                    "type": "person",
                    "name": "Rahul"
                }
            ]
        }

        {
            "detected_entities": [...]
        }

    This deliberately ignores free-form strings.
    """

    candidate_lists = [
        detected_data.get("entities"),
        detected_data.get("detected_entities"),
    ]

    for candidates in candidate_lists:
        if not isinstance(candidates, list):
            continue

        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue

            entity_type = _valid_entity_type(
                candidate.get(
                    "type",
                    candidate.get(
                        "entity_type",
                        "unknown",
                    ),
                )
            )

            if entity_type == "unknown":
                continue

            value = _clean_string(
                candidate.get(
                    "name",
                    candidate.get(
                        "value",
                    ),
                )
            )

            if value is None:
                continue

            normalized = _normalize_identifier(
                value,
                entity_type,
            )

            if normalized is None:
                continue

            key = (
                entity_type,
                normalized,
            )

            candidate_confidence = _safe_confidence(
                candidate.get(
                    "confidence",
                )
            )

            if candidate_confidence is None:
                candidate_confidence = confidence

            entity = registry.get(key)

            if entity is None:
                entity = EntityRecord(
                    entity_id=_new_id(),
                    entity_type=entity_type,
                    name=value,
                    evidence_id=evidence_id,
                    source=source,
                    status=status,
                    confidence=candidate_confidence,
                )

                registry[key] = entity

            else:
                entity.add_evidence(
                    evidence_id=evidence_id,
                    source=source,
                    status=status,
                    confidence=candidate_confidence,
                )

            entity.add_detected_value(value)

            for attribute_name in (
                "role",
                "description",
                "identifier",
                "amount",
                "currency",
            ):
                if attribute_name in candidate:
                    entity.add_attribute(
                        attribute_name,
                        candidate.get(attribute_name),
                    )


# ---------------------------------------------------------------------------
# OBJECT CONVERSION
# ---------------------------------------------------------------------------

def _as_dict(value: Any) -> Dict[str, Any]:
    """
    Convert EvidenceItem/Pydantic/dict/plain object into a dictionary.
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


# ---------------------------------------------------------------------------
# ENTITY LOOKUP HELPERS
# ---------------------------------------------------------------------------

def find_entity(
    entities: Iterable[Dict[str, Any]],
    *,
    entity_type: str,
    name: str,
) -> Optional[Dict[str, Any]]:
    """
    Find an entity conservatively by type and normalized name.
    """

    normalized_type = _valid_entity_type(
        entity_type
    )

    normalized_name = _normalize_identifier(
        name,
        normalized_type,
    )

    if normalized_name is None:
        return None

    for entity in entities:
        if not isinstance(entity, dict):
            continue

        candidate_type = _valid_entity_type(
            entity.get("entity_type")
        )

        if candidate_type != normalized_type:
            continue

        candidate_name = _normalize_identifier(
            entity.get("name"),
            normalized_type,
        )

        if candidate_name == normalized_name:
            return entity

    return None


def group_entities_by_type(
    entities: Iterable[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Group entities by canonical type.

    Useful for downstream relationship and story generation.
    """

    grouped: Dict[str, List[Dict[str, Any]]] = {}

    for entity in entities:
        if not isinstance(entity, dict):
            continue

        entity_type = _valid_entity_type(
            entity.get("entity_type")
        )

        grouped.setdefault(
            entity_type,
            [],
        ).append(
            deepcopy(entity)
        )

    return grouped


# ---------------------------------------------------------------------------
# ENTITY SUMMARY
# ---------------------------------------------------------------------------

def build_entity_summary(
    entities: Iterable[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Build a compact entity summary for the TrustPath case record.

    This is useful for the frontend and future reports.
    """

    entity_list = [
        deepcopy(entity)
        for entity in entities
        if isinstance(entity, dict)
    ]

    grouped = group_entities_by_type(
        entity_list
    )

    return {
        "total": len(entity_list),
        "by_type": {
            entity_type: len(items)
            for entity_type, items in grouped.items()
        },
        "types_present": list(grouped.keys()),
    }


# ---------------------------------------------------------------------------
# TRACEABILITY VALIDATION
# ---------------------------------------------------------------------------

def validate_entity_traceability(
    entities: Iterable[Dict[str, Any]],
    evidence: Iterable[Any],
) -> Dict[str, Any]:
    """
    Validate that every entity points back to at least one evidence item.

    This is important for TrustPath reporting:
        every important story claim should be traceable to evidence.
    """

    evidence_ids = set()

    for item in evidence:
        item_dict = _as_dict(item)

        evidence_id = _clean_string(
            item_dict.get("evidence_id")
        )

        if evidence_id:
            evidence_ids.add(
                evidence_id
            )

    total = 0
    traced = 0
    untraced_ids: List[str] = []

    for entity in entities:
        if not isinstance(entity, dict):
            continue

        total += 1

        entity_id = _clean_string(
            entity.get("entity_id")
        ) or "unknown_entity"

        source_ids = entity.get(
            "source_evidence_ids",
            [],
        )

        if not isinstance(source_ids, list):
            source_ids = []

        valid_sources = [
            source_id
            for source_id in source_ids
            if source_id in evidence_ids
        ]

        if valid_sources:
            traced += 1
        else:
            untraced_ids.append(
                entity_id
            )

    return {
        "total_entities": total,
        "traceable_entities": traced,
        "untraceable_entities": len(
            untraced_ids
        ),
        "untraceable_entity_ids": untraced_ids,
        "valid": len(untraced_ids) == 0,
    }


# ---------------------------------------------------------------------------
# PUBLIC API
# ---------------------------------------------------------------------------

def build_entity_registry(
    evidence: Iterable[Any],
) -> Dict[str, Any]:
    """
    Main public entry point.

    Returns:

        {
            "entities": [...],
            "summary": {...},
            "traceability": {...}
        }

    The returned entities are ready to be mapped into the
    TrustPathEntity Pydantic model.
    """

    evidence_list = list(evidence)

    entities = extract_entities_from_evidence(
        evidence_list
    )

    summary = build_entity_summary(
        entities
    )

    traceability = validate_entity_traceability(
        entities,
        evidence_list,
    )

    return {
        "entities": entities,
        "summary": summary,
        "traceability": traceability,
    }


# ---------------------------------------------------------------------------
# OPTIONAL SINGLE-ENTITY HELPER
# ---------------------------------------------------------------------------

def create_entity_from_explicit_value(
    *,
    entity_type: str,
    name: str,
    evidence_id: str,
    source: str = "system",
    status: str = "unknown",
    confidence: Optional[float] = None,
    attributes: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Create one canonical entity from an explicitly supplied value.

    Useful for future modules such as relationship extraction when they
    discover an entity that must be represented in the common registry.
    """

    normalized_type = _valid_entity_type(
        entity_type
    )

    clean_name = _clean_string(
        name
    )

    entity = EntityRecord(
        entity_id=_new_id(),
        entity_type=normalized_type,
        name=clean_name,
        evidence_id=evidence_id,
        source=source,
        status=_safe_status(status),
        confidence=_safe_confidence(confidence),
    )

    if attributes:
        for key, value in attributes.items():
            entity.add_attribute(
                str(key),
                value,
            )

    return entity.to_dict()


__all__ = [
    "ENTITY_TYPES",
    "EntityRecord",
    "extract_entities_from_evidence",
    "build_entity_registry",
    "build_entity_summary",
    "find_entity",
    "group_entities_by_type",
    "validate_entity_traceability",
    "create_entity_from_explicit_value",
    "apply_user_context",
]