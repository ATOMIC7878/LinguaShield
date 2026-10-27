"""
LinguaShield - TrustPath Router
--------------------------------

Isolated TrustPath backend module.

Purpose:
    - Accept evidence produced by the existing LinguaShield modules.
    - Accept optional user annotations/context.
    - Build a structured TrustPath case record.
    - Keep the record suitable for future PDF/JSON/report exports.
    - Never modify or depend on the internal implementation of the
      existing Text, Visual, Acoustic, Biometric, or Live modules.

Important:
    This file intentionally contains orchestration and data-contract logic
    only. Real detection/model integration should be connected later through
    the adapter layer.

Expected project location:
    LinguaShield/backend/trustpath_router.py
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4
import asyncio

import httpx
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field

# TrustPath's only bridge to the existing analysis modules.
# The fallback keeps compatibility with direct `import trustpath_router`
# when the backend directory itself is on PYTHONPATH.
try:
    from .services.trustpath.adapter import adapt_existing_analysis_dict
except ImportError:
    from services.trustpath.adapter import adapt_existing_analysis_dict


# ---------------------------------------------------------------------------
# ROUTER
# ---------------------------------------------------------------------------

router = APIRouter(
    prefix="/api/trustpath",
    tags=["TrustPath"],
)


# ---------------------------------------------------------------------------
# TYPE DEFINITIONS
# ---------------------------------------------------------------------------

EvidenceSource = Literal[
    "text",
    "visual",
    "acoustic",
    "biometric",
    "user",
    "system",
]

EvidenceStatus = Literal[
    "detected",
    "user_provided",
    "verified",
    "unverified",
    "contradicted",
    "unknown",
]

EntityType = Literal[
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
]

RelationshipStatus = Literal[
    "claimed",
    "inferred",
    "supported",
    "contradicted",
    "verified",
    "unknown",
]

RiskLevel = Literal[
    "LOW",
    "MEDIUM",
    "HIGH",
    "CRITICAL",
    "UNKNOWN",
]


# ---------------------------------------------------------------------------
# REQUEST MODELS
# ---------------------------------------------------------------------------

class EvidenceItem(BaseModel):
    """
    Normalized evidence item.

    This is intentionally generic so existing modules can be adapted to this
    structure later without changing their internal implementations.
    """

    evidence_id: Optional[str] = None

    source: EvidenceSource

    evidence_type: str = Field(
        ...,
        description="Human-readable evidence category, e.g. message, image, transaction."
    )

    title: Optional[str] = None

    description: Optional[str] = None

    content: Optional[str] = Field(
        default=None,
        description="Optional raw text/content associated with the evidence artifact."
    )

    user_context: Optional[str] = Field(
        default=None,
        description="Optional context supplied by the user."
    )

    status: EvidenceStatus = "unknown"

    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    detected_data: Dict[str, Any] = Field(default_factory=dict)

    source_reference: Optional[str] = None

    related_entity_ids: List[str] = Field(default_factory=list)

    related_event_ids: List[str] = Field(default_factory=list)


class UserAnnotation(BaseModel):
    """
    Optional human-provided context.

    User annotations are deliberately separate from detected facts.
    """

    annotation_id: Optional[str] = None

    evidence_id: Optional[str] = None

    label: Optional[str] = None

    context: str

    created_at: Optional[str] = None


class TrustPathInputManifest(BaseModel):
    """Declare which TrustPath input spaces are populated."""

    text: bool = False
    visual: bool = False
    acoustic: bool = False
    reference_voice: bool = False
    test_voice: bool = False

    def provided_sources(self) -> List[str]:
        """Return canonical evidence sources explicitly marked present."""
        sources: List[str] = []
        if self.text:
            sources.append("text")
        if self.visual:
            sources.append("visual")
        if self.acoustic:
            sources.append("acoustic")
        if self.reference_voice or self.test_voice:
            sources.append("biometric")
        return sources

    def biometric_ready(self) -> bool:
        """Biometric comparison is runnable only when both voices exist."""
        return self.reference_voice and self.test_voice


# ---------------------------------------------------------------------------
# OPTIONAL MODULE-BUNDLE INPUT
# ---------------------------------------------------------------------------

class TrustPathModuleBundle(BaseModel):
    """
    Canonical backend bridge for already-produced LinguaShield module outputs.

    Each field accepts the existing JSON/dict returned by its corresponding
    module. TrustPath only normalizes these outputs; it does not rerun them.
    """

    case_name: Optional[str] = None
    user_context: Optional[str] = None
    text: Dict[str, Any] = Field(default_factory=dict)
    visual: Dict[str, Any] = Field(default_factory=dict)
    acoustic: Dict[str, Any] = Field(default_factory=dict)
    biometric: Dict[str, Any] = Field(default_factory=dict)
    evidence: List[Dict[str, Any]] = Field(default_factory=list)
    annotations: List[UserAnnotation] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# INTERNAL RESPONSE MODELS
# ---------------------------------------------------------------------------

class TrustPathEntity(BaseModel):
    entity_id: str

    entity_type: EntityType

    name: Optional[str] = None

    source_evidence_ids: List[str] = Field(default_factory=list)

    attributes: Dict[str, Any] = Field(default_factory=dict)

    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )


class TrustPathRelationship(BaseModel):
    relationship_id: str

    subject_entity_id: str

    relation: str

    object_entity_id: str

    status: RelationshipStatus = "unknown"

    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    evidence_ids: List[str] = Field(default_factory=list)

    explanation: Optional[str] = None


class TrustPathEvent(BaseModel):
    event_id: str

    title: str

    description: str

    event_type: str

    sequence: int

    evidence_ids: List[str] = Field(default_factory=list)

    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )


class TrustPathContradiction(BaseModel):
    contradiction_id: str

    title: str

    description: str

    evidence_ids: List[str] = Field(default_factory=list)

    severity: RiskLevel = "UNKNOWN"


class TrustPathStage(BaseModel):
    stage_id: str

    sequence: int

    title: str

    summary: str

    evidence_ids: List[str] = Field(default_factory=list)

    technique: Optional[str] = None

    attacker_objective: Optional[str] = None

    victim_state: Optional[str] = None

    risk_impact: RiskLevel = "UNKNOWN"

    likely_next_action: Optional[str] = None

    protection: Optional[str] = None


class TrustPathScenario(BaseModel):
    scenario_type: str = "unknown"

    title: str = "Scenario not determined"

    summary: str = "Insufficient evidence to reconstruct a scenario."

    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )


class TrustPathAssessment(BaseModel):
    identity_confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    relationship_confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    scenario_confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    trust_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=100.0,
    )

    risk_level: RiskLevel = "UNKNOWN"


class TrustPathRecommendation(BaseModel):
    action: str

    reason: str

    verification_required: bool = False


class TrustPathStory(BaseModel):
    headline: str

    summary: str

    current_manipulation: Optional[str] = None

    victim_decision: Optional[str] = None

    likely_next_harmful_action: Optional[str] = None

    stages: List[TrustPathStage] = Field(default_factory=list)


class TrustPathCase(BaseModel):
    case_id: str

    created_at: str

    updated_at: str

    status: str

    case_name: Optional[str] = None

    user_context: Optional[str] = None

    input_sources: List[str] = Field(default_factory=list)

    evidence: List[EvidenceItem] = Field(default_factory=list)

    annotations: List[UserAnnotation] = Field(default_factory=list)

    entities: List[TrustPathEntity] = Field(default_factory=list)

    relationships: List[TrustPathRelationship] = Field(default_factory=list)

    events: List[TrustPathEvent] = Field(default_factory=list)

    contradictions: List[TrustPathContradiction] = Field(default_factory=list)

    scenario: TrustPathScenario = Field(
        default_factory=TrustPathScenario
    )

    assessment: TrustPathAssessment = Field(
        default_factory=TrustPathAssessment
    )

    story: TrustPathStory = Field(
        default_factory=lambda: TrustPathStory(
            headline="TrustPath analysis pending",
            summary="No analysis story has been generated yet.",
        )
    )

    recommendation: TrustPathRecommendation = Field(
        default_factory=lambda: TrustPathRecommendation(
            action="No recommendation available yet.",
            reason="Additional evidence is required.",
        )
    )

    report: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# TEMPORARY CASE STORE
# ---------------------------------------------------------------------------
#
# This is intentionally in-memory for the first backend phase.
#
# Later this can be replaced by:
#     - SQLite
#     - PostgreSQL
#     - another persistence layer
#
# without changing the TrustPath response contract.

_CASE_STORE: Dict[str, TrustPathCase] = {}


# ---------------------------------------------------------------------------
# UTILITY FUNCTIONS
# ---------------------------------------------------------------------------

def _utc_now() -> str:
    """Return a timezone-aware ISO timestamp."""
    return datetime.now(timezone.utc).isoformat()


def _new_id(prefix: str) -> str:
    """Create a readable local identifier."""
    return f"{prefix}_{uuid4().hex[:12]}"


def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    """Clamp a numeric value to a safe range."""
    return max(minimum, min(maximum, value))


def _deduplicate_strings(values: List[str]) -> List[str]:
    """Preserve order while removing duplicate strings."""
    seen = set()
    result = []

    for value in values:
        if not value or value in seen:
            continue

        seen.add(value)
        result.append(value)

    return result


# ---------------------------------------------------------------------------
# ADAPTER LAYER
# ---------------------------------------------------------------------------

def normalize_existing_analysis(
    existing_analysis: Dict[str, Any],
    *,
    user_context: Optional[str] = None,
) -> List[EvidenceItem]:
    """
    Convert existing-module results into canonical TrustPath evidence
    through the isolated adapter layer.

    TrustPath does not call or modify the existing detection modules here.
    The adapter only translates their already-produced results.
    """

    adapted = adapt_existing_analysis_dict(
        existing_analysis,
        user_context=user_context,
    )

    normalized: List[EvidenceItem] = []

    for item in adapted.get("evidence", []):
        if not isinstance(item, dict):
            continue

        try:
            normalized.append(
                EvidenceItem(**item)
            )
        except Exception:
            # A malformed optional module result must not break an entire
            # TrustPath case.
            continue

    return normalized

def normalize_evidence(
    payload: TrustPathInput,
) -> List[EvidenceItem]:
    """
    Merge user-supplied evidence and existing-module evidence.

    Existing evidence remains read-only.
    """

    result: List[EvidenceItem] = []

    for item in payload.evidence:
        copied = item.model_copy(deep=True)

        if not copied.evidence_id:
            copied.evidence_id = _new_id("evidence")

        result.append(copied)

    existing_items = normalize_existing_analysis(
        payload.existing_analysis,
        user_context=payload.user_context,
    )

    result.extend(existing_items)

    return result


# ---------------------------------------------------------------------------
# ENTITY EXTRACTION
# ---------------------------------------------------------------------------

def _entity_from_detected_value(
    *,
    value: Any,
    entity_type: EntityType,
    evidence_id: str,
) -> Optional[TrustPathEntity]:
    """
    Create an entity only when a structured value is explicitly present.

    No broad guessing is performed here.
    """

    if value is None:
        return None

    if isinstance(value, str):
        clean_value = value.strip()

        if not clean_value:
            return None

        return TrustPathEntity(
            entity_id=_new_id("entity"),
            entity_type=entity_type,
            name=clean_value,
            source_evidence_ids=[evidence_id],
        )

    return None


def extract_entities(
    evidence: List[EvidenceItem],
) -> List[TrustPathEntity]:
    """
    Extract explicit structured entities from normalized evidence.

    Later this function can be replaced/enhanced with dedicated entity
    extraction while preserving the same output contract.
    """

    entities: List[TrustPathEntity] = []

    for item in evidence:
        data = item.detected_data or {}

        candidate_fields = [
            ("person_name", "person"),
            ("sender_name", "person"),
            ("recipient_name", "person"),
            ("brand_name", "brand"),
            ("company_name", "organization"),
            ("organization_name", "organization"),
            ("phone", "phone"),
            ("phone_number", "phone"),
            ("email", "email"),
            ("email_address", "email"),
            ("transaction_id", "transaction"),
            ("invoice_number", "invoice"),
            ("product_name", "product"),
            ("url", "url"),
        ]

        for field_name, entity_type in candidate_fields:
            value = data.get(field_name)

            entity = _entity_from_detected_value(
                value=value,
                entity_type=entity_type,  # type: ignore[arg-type]
                evidence_id=item.evidence_id or "",
            )

            if entity is not None:
                entity.attributes["detected_field"] = field_name
                entities.append(entity)

    # Add evidence-artifact nodes only when there is evidence to correlate.
    # These nodes represent the supplied artifacts themselves, not real-world
    # people or organizations, so they use the conservative "unknown" type.
    if len(evidence) > 1:
        for item in evidence:
            artifact_entity = TrustPathEntity(
                entity_id=_new_id("entity"),
                entity_type="unknown",
                name=item.title or item.evidence_type,
                source_evidence_ids=[item.evidence_id or ""],
                attributes={
                    "artifact_node": True,
                    "source": item.source,
                    "evidence_type": item.evidence_type,
                },
            )
            entities.append(artifact_entity)

    return entities


# ---------------------------------------------------------------------------
# RELATIONSHIP MAPPING
# ---------------------------------------------------------------------------


def build_relationships(
    entities: List[TrustPathEntity],
    evidence: List[EvidenceItem],
) -> List[TrustPathRelationship]:
    """
    Build conservative relationship edges from explicit metadata plus
    deterministic cross-evidence matches.

    Relationships are created only from observable evidence. Generic "cleared"
    classifications are intentionally excluded from corroboration because
    two clear results do not constitute meaningful threat corroboration.
    """

    entity_ids = {entity.entity_id for entity in entities}
    relationships: List[TrustPathRelationship] = []

    for item in evidence:
        data = item.detected_data or {}
        sender_entity_id = data.get("sender_entity_id")
        recipient_entity_id = data.get("recipient_entity_id")
        relation = data.get("relationship")

        if (
            isinstance(sender_entity_id, str)
            and isinstance(recipient_entity_id, str)
            and isinstance(relation, str)
            and sender_entity_id in entity_ids
            and recipient_entity_id in entity_ids
        ):
            relationships.append(
                TrustPathRelationship(
                    relationship_id=_new_id("relationship"),
                    subject_entity_id=sender_entity_id,
                    relation=relation,
                    object_entity_id=recipient_entity_id,
                    status=data.get("relationship_status", "inferred"),
                    confidence=_safe_optional_float(
                        data.get("relationship_confidence")
                    ),
                    evidence_ids=[item.evidence_id or ""],
                    explanation=data.get("relationship_explanation"),
                )
            )

    artifact_entities = {
        entity.source_evidence_ids[0]: entity
        for entity in entities
        if entity.attributes.get("artifact_node") is True
        and entity.source_evidence_ids
    }

    def _norm(value: Any) -> Optional[str]:
        if not isinstance(value, str):
            return None
        import re
        lowered = value.lower().strip()
        lowered = re.sub(r"[^\w\s]", " ", lowered, flags=re.UNICODE)
        return " ".join(lowered.split()) or None

    def _artifact_text(item: EvidenceItem) -> Optional[str]:
        data = item.detected_data or {}
        return _norm(
            data.get("extracted_text")
            or data.get("processed_text")
            or data.get("ocr")
            or item.content
        )

    def _artifact_reference(item: EvidenceItem) -> Optional[str]:
        data = item.detected_data or {}
        return _norm(data.get("source_file") or item.source_reference)

    def _voice_target_reference(item: EvidenceItem) -> Optional[str]:
        data = item.detected_data or {}
        return _norm(data.get("trustpath_reference_target_id"))

    def _scenario_type(item: EvidenceItem) -> Optional[str]:
        return _norm((item.detected_data or {}).get("scenario_type"))

    def _risk_classification(item: EvidenceItem) -> Optional[str]:
        return _norm((item.detected_data or {}).get("risk_classification"))

    def _bool_signal(item: EvidenceItem, *keys: str) -> Optional[bool]:
        data = item.detected_data or {}
        for key in keys:
            value = data.get(key)
            if isinstance(value, bool):
                return value
        return None

    seen_pair_relations = set()

    def _add_relationship(
        *,
        left_entity: TrustPathEntity,
        right_entity: TrustPathEntity,
        relation: str,
        confidence: Optional[float],
        evidence_ids: List[str],
        explanation: str,
        status: RelationshipStatus = "inferred",
    ) -> None:
        pair_key = tuple(sorted((left_entity.entity_id, right_entity.entity_id)))
        relation_key = (pair_key, relation)
        if relation_key in seen_pair_relations:
            return
        seen_pair_relations.add(relation_key)
        relationships.append(
            TrustPathRelationship(
                relationship_id=_new_id("relationship"),
                subject_entity_id=left_entity.entity_id,
                relation=relation,
                object_entity_id=right_entity.entity_id,
                status=status,
                confidence=confidence,
                evidence_ids=evidence_ids,
                explanation=explanation,
            )
        )

    for index, left in enumerate(evidence):
        left_entity = artifact_entities.get(left.evidence_id)
        if left_entity is None:
            continue

        left_text = _artifact_text(left)
        left_ref = _artifact_reference(left)
        left_voice_ref = _voice_target_reference(left)
        left_scenario = _scenario_type(left)
        left_risk = _risk_classification(left)
        left_threat = _bool_signal(left, "threat_detected", "is_scam")

        for right in evidence[index + 1:]:
            right_entity = artifact_entities.get(right.evidence_id)
            if right_entity is None:
                continue

            right_text = _artifact_text(right)
            right_ref = _artifact_reference(right)
            right_voice_ref = _voice_target_reference(right)
            right_scenario = _scenario_type(right)
            right_risk = _risk_classification(right)
            right_threat = _bool_signal(right, "threat_detected", "is_scam")

            if left_text and right_text and left_text == right_text:
                _add_relationship(
                    left_entity=left_entity,
                    right_entity=right_entity,
                    relation="same_extracted_content_as",
                    confidence=1.0,
                    evidence_ids=[left.evidence_id or "", right.evidence_id or ""],
                    explanation=(
                        "The supplied evidence artifacts contain the same "
                        "normalized extracted content."
                    ),
                )

            elif (
                left_voice_ref
                and right_voice_ref
                and left_voice_ref == right_voice_ref
            ):
                _add_relationship(
                    left_entity=left_entity,
                    right_entity=right_entity,
                    relation="same_trustpath_voice_reference_as",
                    confidence=1.0,
                    evidence_ids=[left.evidence_id or "", right.evidence_id or ""],
                    explanation=(
                        "The supplied biometric and voice-analysis evidence "
                        "share the same TrustPath reference target."
                    ),
                )

            elif left_ref and right_ref and left_ref == right_ref:
                _add_relationship(
                    left_entity=left_entity,
                    right_entity=right_entity,
                    relation="same_source_artifact_as",
                    confidence=1.0,
                    evidence_ids=[left.evidence_id or "", right.evidence_id or ""],
                    explanation=(
                        "The supplied evidence artifacts reference the same "
                        "source file or recording."
                    ),
                )

            elif left_scenario and right_scenario and left_scenario == right_scenario:
                _add_relationship(
                    left_entity=left_entity,
                    right_entity=right_entity,
                    relation="supports",
                    confidence=1.0,
                    evidence_ids=[left.evidence_id or "", right.evidence_id or ""],
                    explanation=(
                        "Both evidence sources explicitly report the same "
                        "scenario type."
                    ),
                )

            else:
                # Conservative semantic same-claim correlation. Only Text ↔ Visual
                # is eligible for this inferred relation because those modalities
                # represent independently supplied claim artifacts. Acoustic
                # transcripts and biometric transcripts can share words with a
                # message without proving that they express the same claim.
                semantic_pair_allowed = {
                    left.source,
                    right.source,
                } == {"text", "visual"}

                if semantic_pair_allowed:
                    left_claim_tokens = _semantic_claim_tokens(left)
                    right_claim_tokens = _semantic_claim_tokens(right)
                    shared_terms = left_claim_tokens.intersection(right_claim_tokens)
                    left_context = _contextual_claim_signals(left)
                    right_context = _contextual_claim_signals(right)

                    if (
                        len(shared_terms) >= 2
                        and left_context["present"]
                        and right_context["present"]
                    ):
                        overlap_confidence = _clamp(
                            min(0.95, 0.65 + (0.05 * min(len(shared_terms), 6)))
                        )
                        _add_relationship(
                            left_entity=left_entity,
                            right_entity=right_entity,
                            relation="same_claim_as",
                            confidence=overlap_confidence,
                            evidence_ids=[left.evidence_id or "", right.evidence_id or ""],
                            explanation=(
                                "The Text and Visual artifacts contain overlapping, "
                                "non-generic terms and independent contextual signals "
                                "indicating the same underlying claim or request context."
                            ),
                        )

            if (
                left_risk
                and right_risk
                and left_risk == right_risk
                and _relationship_is_meaningful_for_correlation(left, right)
                and not (
                    left_scenario
                    and right_scenario
                    and left_scenario == right_scenario
                )
            ):
                _add_relationship(
                    left_entity=left_entity,
                    right_entity=right_entity,
                    relation="corroborates",
                    confidence=1.0,
                    evidence_ids=[left.evidence_id or "", right.evidence_id or ""],
                    explanation=(
                        "Both evidence sources explicitly report the same "
                        "meaningful risk classification."
                    ),
                )

            # Conflict is derived from explicit content-threat disagreement only.
            # A raw is_scam/threat_detected mismatch is NOT enough when:
            #   1) both artifacts explicitly identify the same scenario, or
            #   2) either artifact's classification is a voice-origin finding
            #      (for example REVIEW · AI VOICE), which is a different analytical
            #      dimension rather than a content-safety verdict.
            def _voice_origin_only(item: EvidenceItem) -> bool:
                classification = str(
                    (item.detected_data or {}).get("risk_classification") or ""
                ).strip().upper()
                scenario = str(
                    (item.detected_data or {}).get("scenario_type") or ""
                ).strip().lower()
                voice_markers = (
                    "AI VOICE",
                    "HUMAN VOICE",
                    "INCONCLUSIVE VOICE",
                    "REVIEW · AI VOICE",
                    "REVIEW · INCONCLUSIVE VOICE",
                )
                return (
                    any(marker in classification for marker in voice_markers)
                    or scenario.startswith("review_")
                    or scenario.startswith("human_voice")
                    or scenario.startswith("ai_voice")
                    or scenario.startswith("voice_identity_")
                )

            same_explicit_scenario = bool(
                left_scenario
                and right_scenario
                and left_scenario == right_scenario
            )
            voice_origin_pair = _voice_origin_only(left) or _voice_origin_only(right)

            if (
                left_threat is not None
                and right_threat is not None
                and left_threat != right_threat
                and not same_explicit_scenario
                and not voice_origin_pair
            ):
                _add_relationship(
                    left_entity=left_entity,
                    right_entity=right_entity,
                    relation="contradicts",
                    confidence=1.0,
                    evidence_ids=[left.evidence_id or "", right.evidence_id or ""],
                    explanation=(
                        "The evidence sources contain conflicting explicit "
                        "content-threat verdicts."
                    ),
                    status="contradicted",
                )

    return relationships



def _safe_optional_float(value: Any) -> Optional[float]:
    """Safely convert an optional value to a 0..1 float."""
    if value is None:
        return None

    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None

    return _clamp(converted)


def _biometric_score_to_unit(value: Any) -> Optional[float]:
    """Normalize a biometric score that may be expressed as 0..1 or 0..100."""
    if value is None:
        return None

    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None

    if 0.0 <= converted <= 1.0:
        return converted
    if 0.0 <= converted <= 100.0:
        return converted / 100.0
    return None


def _biometric_identity_status(
    *,
    match: Optional[bool],
    similarity: Optional[float],
    impersonation_risk: Optional[float],
) -> Dict[str, Any]:
    """Interpret explicit biometric identity signals without inventing identity."""
    if match is True:
        return {
            "status": "verified",
            "confidence": similarity,
            "risk": "LOW",
        }

    if match is False:
        risk = "HIGH"
        if impersonation_risk is not None:
            if impersonation_risk >= 0.70:
                risk = "HIGH"
            elif impersonation_risk >= 0.45:
                risk = "MEDIUM"
            else:
                risk = "LOW"
        return {
            "status": "mismatch",
            "confidence": similarity,
            "risk": risk,
        }

    return {
        "status": "unknown",
        "confidence": similarity,
        "risk": "UNKNOWN",
    }


# ---------------------------------------------------------------------------
# EVENT EXTRACTION
# ---------------------------------------------------------------------------

def build_events(
    evidence: List[EvidenceItem],
) -> List[TrustPathEvent]:
    """
    Convert explicit event-like evidence into ordered events.

    Biometric mismatch/impersonation signals are converted into a dedicated
    evidence event because they are explicit upstream verification findings.
    This intentionally does not invent chronology.
    """

    events: List[TrustPathEvent] = []

    for item in evidence:
        data = item.detected_data or {}

        event_title = data.get("event_title")
        event_description = data.get("event_description")
        event_type = data.get("event_type")

        if item.source == "biometric" and not event_title and not event_description:
            match = data.get("match")
            if match is None:
                match = data.get("verified")
            if match is None:
                match = data.get("biometric_verified")

            similarity = _biometric_score_to_unit(
                data.get("similarity", data.get("similarity_score"))
            )
            impersonation_risk = _biometric_score_to_unit(
                data.get("impersonation_risk")
            )
            classification_label = data.get("classification_label")

            label_upper = (
                classification_label.upper()
                if isinstance(classification_label, str)
                else ""
            )
            mismatch = match is False
            explicit_impersonation = (
                mismatch
                or "IMPERSONATION" in label_upper
                or "MISMATCH" in label_upper
                or (
                    impersonation_risk is not None
                    and impersonation_risk >= 0.50
                )
            )

            if explicit_impersonation:
                event_title = "Biometric identity verification"
                if mismatch:
                    event_description = (
                        "The test voice does not sufficiently match the "
                        "supplied reference voice."
                    )
                elif classification_label:
                    event_description = str(classification_label)
                else:
                    event_description = "Explicit biometric impersonation risk detected."
                event_type = "biometric_verification"
            elif match is True:
                event_title = "Biometric identity verification"
                if similarity is not None:
                    event_description = (
                        "The supplied test voice matches the reference voice "
                        f"with {similarity * 100:.2f}% similarity."
                    )
                else:
                    event_description = (
                        "The supplied test voice matches the reference voice."
                    )
                event_type = "biometric_verification"

        if not event_title and not event_description:
            continue

        events.append(
            TrustPathEvent(
                event_id=_new_id("event"),
                title=str(
                    event_title or item.title or "Observed event"
                ),
                description=str(
                    event_description or item.description or ""
                ),
                event_type=str(event_type or "observed"),
                sequence=len(events) + 1,
                evidence_ids=[item.evidence_id or ""],
                confidence=_safe_optional_float(
                    data.get("event_confidence")
                ),
            )
        )

    return events


def detect_contradictions(
    evidence: List[EvidenceItem],
    relationships: List[TrustPathRelationship],
) -> List[TrustPathContradiction]:
    """
    Detect explicit and deterministic evidence contradictions.

    Contradictions are limited to:
        - explicit upstream contradiction metadata,
        - directly conflicting explicit threat booleans, or
        - a deterministic ``contradicts`` relationship.
    """

    contradictions: List[TrustPathContradiction] = []
    seen_keys = set()

    def _add(
        *,
        title: str,
        description: str,
        evidence_ids: List[str],
        severity: RiskLevel,
    ) -> None:
        clean_ids = _deduplicate_strings(evidence_ids)
        key = (tuple(sorted(clean_ids)), title)
        if key in seen_keys:
            return
        seen_keys.add(key)
        contradictions.append(
            TrustPathContradiction(
                contradiction_id=_new_id("contradiction"),
                title=title,
                description=description,
                evidence_ids=clean_ids,
                severity=severity,
            )
        )

    for item in evidence:
        data = item.detected_data or {}
        contradiction_title = data.get("contradiction_title")
        contradiction_description = data.get("contradiction_description")
        if not contradiction_title and not contradiction_description:
            continue

        severity = data.get("contradiction_severity", "UNKNOWN")
        if severity not in {"LOW", "MEDIUM", "HIGH", "CRITICAL", "UNKNOWN"}:
            severity = "UNKNOWN"

        _add(
            title=str(contradiction_title or "Evidence contradiction"),
            description=str(contradiction_description or ""),
            evidence_ids=[item.evidence_id or ""],
            severity=severity,
        )

    for relationship in relationships:
        if relationship.relation != "contradicts":
            continue
        _add(
            title="Conflicting evidence verdict",
            description=(
                relationship.explanation
                or "Two supplied evidence sources contain conflicting explicit verdicts."
            ),
            evidence_ids=relationship.evidence_ids,
            severity="HIGH",
        )

    return contradictions



# ---------------------------------------------------------------------------
# SCENARIO RECONSTRUCTION
# ---------------------------------------------------------------------------

def _upstream_signals(evidence: List[EvidenceItem]) -> Dict[str, Any]:
    """Collect explicit upstream verdict fields without inventing new facts."""
    signals: Dict[str, Any] = {
        "is_scam": None,
        "threat_detected": None,
        "risk_classification": None,
        "confidence": None,
        "has_explicit_scenario": False,
    }

    for item in evidence:
        data = item.detected_data or {}

        if data.get("is_scam") is not None and signals["is_scam"] is None:
            signals["is_scam"] = bool(data.get("is_scam"))

        if data.get("threat_detected") is not None and signals["threat_detected"] is None:
            signals["threat_detected"] = bool(data.get("threat_detected"))

        risk = data.get("risk_classification")
        if isinstance(risk, str) and risk.strip() and signals["risk_classification"] is None:
            signals["risk_classification"] = risk.strip()

        confidence = _safe_optional_float(data.get("confidence"))
        if confidence is not None and signals["confidence"] is None:
            signals["confidence"] = confidence

        candidate = data.get("scenario_type")
        if isinstance(candidate, str) and candidate.strip():
            signals["has_explicit_scenario"] = True

    return signals


def _biometric_signals(evidence: List[EvidenceItem]) -> Dict[str, Any]:
    """Collect explicit biometric identity/impersonation signals."""
    signals: Dict[str, Any] = {
        "present": False,
        "match": None,
        "similarity": None,
        "impersonation_risk": None,
        "classification_label": None,
    }

    for item in evidence:
        if item.source != "biometric":
            continue

        signals["present"] = True
        data = item.detected_data or {}

        if signals["match"] is None:
            match = data.get("match")
            if match is None:
                match = data.get("verified")
            if match is None:
                match = data.get("biometric_verified")
            if match is not None:
                signals["match"] = bool(match)

        if signals["similarity"] is None:
            similarity = _biometric_score_to_unit(
                data.get("similarity", data.get("similarity_score"))
            )
            if similarity is not None:
                signals["similarity"] = similarity

        if signals["impersonation_risk"] is None:
            impersonation_risk = _biometric_score_to_unit(
                data.get("impersonation_risk")
            )
            if impersonation_risk is not None:
                signals["impersonation_risk"] = impersonation_risk

        label = data.get("classification_label")
        if (
            isinstance(label, str)
            and label.strip()
            and signals["classification_label"] is None
        ):
            signals["classification_label"] = label.strip()

    return signals

def _classification_to_risk(risk_classification: Any) -> RiskLevel:
    """Map only explicit upstream classification wording to a TrustPath risk."""
    if not isinstance(risk_classification, str):
        return "UNKNOWN"

    value = risk_classification.strip().upper()
    if not value:
        return "UNKNOWN"

    if "CRITICAL" in value:
        return "CRITICAL"
    if "HIGH" in value:
        return "HIGH"
    if "MEDIUM" in value or "MODERATE" in value or "SUSPICIOUS" in value:
        return "MEDIUM"
    if "LOW" in value:
        return "LOW"
    if "CLEAR" in value or "BENIGN" in value or "SAFE" in value or "NO THREAT" in value:
        return "LOW"

    return "UNKNOWN"

def _risk_from_evidence_items(
    evidence_items: List[EvidenceItem],
) -> RiskLevel:
    """Return the strongest explicit risk attributable to the supplied evidence."""
    severity_rank = {
        "UNKNOWN": 0,
        "LOW": 1,
        "MEDIUM": 2,
        "HIGH": 3,
        "CRITICAL": 4,
    }

    highest: RiskLevel = "UNKNOWN"
    highest_rank = 0

    for item in evidence_items:
        data = item.detected_data or {}

        # Explicit upstream classification is authoritative for that artifact.
        classification_risk = _classification_to_risk(
            data.get("risk_classification")
        )
        rank = severity_rank[classification_risk]
        if rank > highest_rank:
            highest = classification_risk
            highest_rank = rank

        # Explicit boolean threat verdicts can establish risk even if a
        # classification label is absent.
        if data.get("is_scam") is True or data.get("threat_detected") is True:
            confidence = _safe_optional_float(data.get("confidence")) or 0.0
            bool_risk: RiskLevel = (
                "CRITICAL" if confidence >= 0.90
                else "HIGH" if confidence >= 0.75
                else "MEDIUM" if confidence >= 0.50
                else "LOW"
            )
            rank = severity_rank[bool_risk]
            if rank > highest_rank:
                highest = bool_risk
                highest_rank = rank

        # Biometric mismatch is identity-risk evidence for the biometric
        # artifact only; a verified match is explicitly LOW risk.
        if item.source == "biometric":
            biometric = _biometric_identity_status(
                match=data.get("match", data.get("verified", data.get("biometric_verified"))),
                similarity=_biometric_score_to_unit(
                    data.get("similarity", data.get("similarity_score"))
                ),
                impersonation_risk=_biometric_score_to_unit(
                    data.get("impersonation_risk")
                ),
            )
            biometric_risk = biometric["risk"]
            rank = severity_rank[biometric_risk]
            if rank > highest_rank:
                highest = biometric_risk
                highest_rank = rank

    return highest


def _relationship_is_meaningful_for_correlation(
    left: EvidenceItem,
    right: EvidenceItem,
) -> bool:
    """Exclude generic cleared/benign classifications from threat corroboration."""
    left_risk = _classification_to_risk((left.detected_data or {}).get("risk_classification"))
    right_risk = _classification_to_risk((right.detected_data or {}).get("risk_classification"))

    meaningful_markers = (
        "CLEARED",
        "BENIGN",
        "SAFE",
        "NO THREAT",
    )
    left_label = str((left.detected_data or {}).get("risk_classification") or "").upper()
    right_label = str((right.detected_data or {}).get("risk_classification") or "").upper()

    if left_risk == "LOW" and right_risk == "LOW":
        if any(marker in left_label for marker in meaningful_markers) and any(
            marker in right_label for marker in meaningful_markers
        ):
            return False

    return True


def _contextual_claim_signals(item: EvidenceItem) -> Dict[str, Any]:
    """
    Derive conservative TrustPath contextual signals from supplied evidence
    content and explicit module fields.

    These are advisory observations, not replacements for upstream model
    verdicts. They help TrustPath avoid treating a clearly suspicious claim
    pattern as "safe" merely because an upstream module applied a context
    override.
    """
    data = item.detected_data or {}
    raw_parts = [
        item.content,
        data.get("processed_text"),
        data.get("extracted_text"),
        data.get("ocr"),
        data.get("text"),
    ]
    text = " ".join(
        str(value).lower()
        for value in raw_parts
        if isinstance(value, str) and value.strip()
    )

    import re

    normalized = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = set(normalized.split())

    category_terms = {
        "authority_claim": {
            "bank", "account", "manager", "security", "official",
            "government", "police", "customer", "executive",
        },
        "account_security_claim": {
            "account", "verification", "verify", "identity", "security",
            "flagged", "suspicious", "credential", "login",
        },
        "urgency_pressure": {
            "immediate", "immediately", "urgent", "urgently", "now",
            "today", "quickly", "asap",
        },
        "action_request": {
            "verify", "verification", "confirm", "click", "submit",
            "provide", "send", "share", "code", "otp", "payment",
            "transfer",
        },
        "consequence_threat": {
            "suspension", "suspended", "blocked", "freeze", "frozen",
            "closure", "closed", "penalty", "flagged", "prevent",
        },
    }

    matched_categories = [
        category
        for category, terms in category_terms.items()
        if tokens.intersection(terms)
    ]

    # Explicit model outputs remain authoritative. This heuristic only adds
    # a contextual concern when multiple concrete signals co-occur.
    contextual_concern = (
        len(matched_categories) >= 3
        or (
            "urgency_pressure" in matched_categories
            and "action_request" in matched_categories
        )
        or (
            "authority_claim" in matched_categories
            and "action_request" in matched_categories
            and "consequence_threat" in matched_categories
        )
    )

    severity: RiskLevel = "UNKNOWN"
    if contextual_concern:
        # Contextual heuristics surface concern conservatively. They do not
        # independently escalate to HIGH/CRITICAL because upstream models may
        # have legitimate context-specific reasons for their verdict.
        severity = "MEDIUM"

    return {
        "present": contextual_concern,
        "categories": matched_categories,
        "risk_level": severity,
        "normalized_terms": sorted(tokens),
        "source": item.source,
    }


def _semantic_claim_tokens(item: EvidenceItem) -> set:
    """Return non-generic terms useful for conservative same-claim matching."""
    # Biometric content is an identity-verification artifact, not an independent
    # claim artifact. Its transcript must not create semantic claim edges merely
    # because it shares words with text, visual OCR, or acoustic transcripts.
    if item.source == "biometric":
        return set()

    signals = _contextual_claim_signals(item)
    categories = signals.get("categories") or []
    if not categories:
        return set()

    data = item.detected_data or {}
    raw_parts = [
        item.content,
        data.get("processed_text"),
        data.get("extracted_text"),
        data.get("ocr"),
        data.get("text"),
        data.get("scenario_type"),
    ]
    import re
    text = " ".join(
        str(value).lower()
        for value in raw_parts
        if isinstance(value, str) and value.strip()
    )
    terms = {
        token
        for token in re.sub(r"[^a-z0-9\s]", " ", text).split()
        if len(token) >= 4
    }

    generic = {
        "this", "that", "with", "your", "from", "they", "have", "been",
        "will", "into", "used", "regarding", "customer", "there", "their",
        "analysis", "recording", "evidence", "provided",
    }
    return terms - generic


def _enrich_contextual_signals(evidence: List[EvidenceItem]) -> None:
    """
    Attach TrustPath-only contextual observations to normalized evidence.

    Existing upstream fields are preserved. These fields explicitly identify
    themselves as TrustPath observations so callers can distinguish them from
    module verdicts.
    """
    for item in evidence:
        # Biometric evidence is an identity-verification artifact. Its transcript
        # may contain contextual words such as "account", "verification", or
        # "manager", but those words must not create a TrustPath threat concern
        # for an otherwise verified biometric match. Keep contextual threat
        # heuristics scoped to content-bearing threat modalities.
        if item.source == "biometric":
            continue

        signals = _contextual_claim_signals(item)
        if signals["present"] or signals["categories"]:
            item.detected_data.setdefault(
                "trustpath_contextual_signals",
                {
                    "concern_present": bool(signals["present"]),
                    "risk_level": signals["risk_level"],
                    "categories": signals["categories"],
                    "basis": (
                        "Deterministic contextual pattern across supplied "
                        "evidence content; not an override of the upstream "
                        "module verdict."
                    ),
                },
            )



def _contextual_scenario_from_evidence(
    evidence: List[EvidenceItem],
    relationships: List[TrustPathRelationship],
) -> Optional[TrustPathScenario]:
    """Reconstruct a conservative multimodal scenario from contextual signals.

    This path is intentionally separate from upstream model verdicts. It only
    fires when supplied evidence contains multiple concrete contextual signals
    and, where available, cross-evidence support. It does not relabel an
    upstream module verdict as a threat.
    """
    if not evidence:
        return None

    concerned = []
    category_union = set()
    for item in evidence:
        signals = (item.detected_data or {}).get("trustpath_contextual_signals")
        if isinstance(signals, dict) and signals.get("concern_present"):
            concerned.append(item)
            category_union.update(signals.get("categories") or [])

    # Require at least one contextual concern and either multiple independent
    # evidence items or a concrete relationship connecting the evidence.
    if not concerned:
        return None
    relevant_relationships = [
        relationship
        for relationship in relationships
        if relationship.relation in {"same_claim_as", "supports", "corroborates", "same_extracted_content_as"}
    ]
    if len(evidence) < 2 and not relevant_relationships:
        return None

    if {
        "account_security_claim",
        "urgency_pressure",
    }.issubset(category_union) or {
        "account_security_claim",
        "consequence_threat",
    }.issubset(category_union):
        scenario_type = "suspicious_account_verification"
        title = "Suspicious Account Verification"
        summary = (
            "The supplied evidence describes an account or identity verification "
            "request combined with urgency or a consequence such as suspension. "
            "This is a contextual concern and does not override the upstream module verdicts."
        )
    elif {
        "authority_claim",
        "action_request",
        "consequence_threat",
    }.issubset(category_union):
        scenario_type = "authority_based_action_request"
        title = "Authority-Based Action Request"
        summary = (
            "The supplied evidence combines an authority claim, an action request, "
            "and a stated consequence. This pattern warrants independent verification."
        )
    elif len(category_union) >= 2:
        scenario_type = "contextual_social_engineering"
        title = "Contextual Social-Engineering Concern"
        summary = (
            "Multiple supplied evidence artifacts contain contextual signals associated "
            "with an urgent or authority-based request. Independent verification is recommended."
        )
    else:
        return None

    relationship_confidences = [
        relationship.confidence
        for relationship in relevant_relationships
        if relationship.confidence is not None
    ]
    relationship_confidence = max(relationship_confidences) if relationship_confidences else 0.0

    # The scenario confidence expresses confidence in the reconstructed scenario,
    # not safety or trustworthiness. Contextual evidence is deliberately kept
    # below the certainty of an explicit upstream scenario.
    confidence = 0.65 + (0.20 * relationship_confidence)
    if len(concerned) >= 2:
        confidence += 0.05
    confidence = _clamp(confidence, 0.0, 0.90)

    return TrustPathScenario(
        scenario_type=scenario_type,
        title=title,
        summary=summary,
        confidence=confidence,
    )


def reconstruct_scenario(
    evidence: List[EvidenceItem],
    events: List[TrustPathEvent],
    contradictions: List[TrustPathContradiction],
    relationships: Optional[List[TrustPathRelationship]] = None,
) -> TrustPathScenario:
    """
    Reconstruct the most defensible scenario from explicit evidence.

    Severe contradictions take precedence because they directly affect the
    reliability of any otherwise-supported scenario.
    """

    scenario_type = "unknown"
    title = "Scenario not determined"
    summary = "Insufficient evidence to reconstruct a scenario."
    confidence = 0.0

    # A contradiction is itself a scenario-level finding and must not be hidden
    # simply because one module also supplied a scenario label.
    if contradictions:
        scenario_type = "evidence_inconsistency"
        title = "Evidence inconsistency"
        summary = (
            "The supplied evidence contains one or more conflicting explicit "
            "verdicts that require independent verification."
        )
        severity_rank = {"UNKNOWN": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}
        highest = max(
            (severity_rank.get(item.severity, 0) for item in contradictions),
            default=0,
        )
        confidence = 0.90 if highest >= 4 else 0.80 if highest >= 3 else 0.70
    else:
        explicit_types: List[str] = []

        for item in evidence:
            candidate = (item.detected_data or {}).get("scenario_type")
            if isinstance(candidate, str) and candidate.strip():
                explicit_types.append(candidate.strip())

        if explicit_types:
            counts: Dict[str, int] = {}
            for candidate in explicit_types:
                counts[candidate] = counts.get(candidate, 0) + 1

            scenario_type = max(
                counts,
                key=lambda candidate: (counts[candidate], candidate),
            )
            matching_items = [
                item for item in evidence
                if item.detected_data.get("scenario_type") == scenario_type
            ]

            title = scenario_type.replace("_", " ").title()

            if matching_items:
                summary = (
                    matching_items[0].detected_data.get("scenario_summary")
                    or matching_items[0].description
                    or f"TrustPath identified a {title.lower()} pattern."
                )

            converted = [
                _safe_optional_float(item.detected_data.get("scenario_confidence"))
                for item in matching_items
            ]
            converted = [value for value in converted if value is not None]

            if converted:
                confidence = max(converted)
            elif len(matching_items) > 1:
                confidence = 1.0
            else:
                confidence = 0.70
        else:
            biometric = _biometric_signals(evidence)
            label_upper = str(biometric["classification_label"] or "").upper()

            impersonation_signal = (
                biometric["match"] is False
                or (
                    biometric["impersonation_risk"] is not None
                    and biometric["impersonation_risk"] >= 0.50
                )
                or "IMPERSONATION" in label_upper
                or "MISMATCH" in label_upper
            )

            identity_status = _biometric_identity_status(
                match=biometric["match"],
                similarity=biometric["similarity"],
                impersonation_risk=biometric["impersonation_risk"],
            )

            if (
                biometric["present"]
                and identity_status["status"] == "verified"
                and not impersonation_signal
            ):
                scenario_type = "voice_identity_verified"
                title = "Voice Identity Verified"
                if biometric["similarity"] is not None:
                    summary = (
                        "The supplied test voice matches the reference voice, "
                        f"with {biometric['similarity'] * 100:.2f}% similarity. "
                        "The claimed voice identity is consistent with the "
                        "supplied reference sample."
                    )
                else:
                    summary = (
                        "The supplied test voice matches the reference voice, "
                        "indicating that the claimed voice identity is "
                        "consistent with the supplied reference sample."
                    )
                confidence = biometric["similarity"] or 0.0

            elif biometric["present"] and impersonation_signal:
                scenario_type = "voice_impersonation"
                title = "Voice Impersonation"
                if biometric["match"] is False:
                    summary = (
                        "The supplied test voice does not sufficiently match "
                        "the reference voice, indicating possible voice impersonation."
                    )
                else:
                    summary = (
                        "The biometric evidence contains explicit "
                        "impersonation risk indicators."
                    )
                confidence = (
                    biometric["impersonation_risk"]
                    if biometric["impersonation_risk"] is not None
                    else 0.0
                )
            else:
                upstream = _upstream_signals(evidence)
                malicious = (
                    upstream["is_scam"] is True
                    or upstream["threat_detected"] is True
                )
                classification = upstream["risk_classification"]
                mapped_risk = _classification_to_risk(classification)

                if malicious:
                    scenario_type = "upstream_threat_detection"
                    title = "Upstream threat detection"
                    summary = str(
                        classification
                        or "The existing analysis explicitly detected a threat."
                    )
                    confidence = upstream["confidence"] or 0.0
                elif mapped_risk in {"HIGH", "CRITICAL"}:
                    scenario_type = "upstream_threat_classification"
                    title = "Upstream threat classification"
                    summary = str(classification)
                    confidence = upstream["confidence"] or 0.0
                else:
                    contextual_scenario = _contextual_scenario_from_evidence(
                        evidence,
                        relationships or [],
                    )
                    if contextual_scenario is not None:
                        return contextual_scenario

    return TrustPathScenario(
        scenario_type=scenario_type,
        title=title,
        summary=str(summary),
        confidence=_clamp(confidence),
    )



def build_assessment(
    *,
    evidence: List[EvidenceItem],
    relationships: List[TrustPathRelationship],
    scenario: TrustPathScenario,
    contradictions: List[TrustPathContradiction],
) -> TrustPathAssessment:
    """
    Build an evidence-led TrustPath assessment.

    Trust score represents confidence in the overall conclusion, not confidence
    in the relationship graph. Relationship confidence is reported separately.

    Explicit upstream risk/verdict fields are treated as the primary threat
    signal. Biometric mismatch/impersonation signals are also treated as
    explicit identity-risk evidence. A scenario's existence or confidence
    alone never upgrades risk.
    """
    identity_candidates: List[float] = []
    relationship_candidates: List[float] = []

    for item in evidence:
        data = item.detected_data or {}

        identity_value = _safe_optional_float(data.get("identity_confidence"))
        if identity_value is not None:
            identity_candidates.append(identity_value)

        # A biometric similarity score is an explicit identity signal when
        # the biometric adapter does not provide a separate identity_confidence.
        if item.source == "biometric":
            similarity_value = _biometric_score_to_unit(
                data.get("similarity", data.get("similarity_score"))
            )
            if similarity_value is not None:
                identity_candidates.append(similarity_value)

        relationship_value = _safe_optional_float(data.get("relationship_confidence"))
        if relationship_value is not None:
            relationship_candidates.append(relationship_value)

    biometric = _biometric_signals(evidence)
    if biometric["present"] and biometric["similarity"] is not None:
        # When biometric evidence is explicitly supplied, its similarity is the
        # authoritative identity-confidence signal for this case.
        identity_confidence = biometric["similarity"]
    else:
        identity_confidence = max(identity_candidates) if identity_candidates else None

    if relationship_candidates:
        relationship_confidence = max(relationship_candidates)
    elif relationships:
        # "contradicts" is negative evidence, not positive relationship confidence.
        supportive_relationships = [
            relationship
            for relationship in relationships
            if relationship.relation != "contradicts"
        ]
        relationship_values = [
            relationship.confidence
            for relationship in supportive_relationships
            if relationship.confidence is not None
        ]
        relationship_confidence = max(relationship_values) if relationship_values else None
    else:
        relationship_confidence = None

    risk_level: RiskLevel = "UNKNOWN"

    # Explicit contradictions remain the strongest TrustPath risk signal.
    if contradictions:
        severities = {contradiction.severity for contradiction in contradictions}
        if "CRITICAL" in severities:
            risk_level = "CRITICAL"
        elif "HIGH" in severities:
            risk_level = "HIGH"
        elif "MEDIUM" in severities:
            risk_level = "MEDIUM"
        elif "LOW" in severities:
            risk_level = "LOW"

    # Otherwise aggregate explicit upstream findings across ALL supplied
    # evidence items. A cleared Text/Visual finding must not mask a higher
    # severity Acoustic/other finding that belongs to the same TrustPath case.
    if risk_level == "UNKNOWN":
        severity_rank = {
            "UNKNOWN": 0,
            "LOW": 1,
            "MEDIUM": 2,
            "HIGH": 3,
            "CRITICAL": 4,
        }

        observed_risk: RiskLevel = "UNKNOWN"
        highest_rank = 0
        highest_confidence = 0.0
        explicit_threat_present = False
        explicit_risk_present = False

        for item in evidence:
            data = item.detected_data or {}

            classification = data.get("risk_classification")
            classification_risk = _classification_to_risk(classification)
            if classification_risk != "UNKNOWN":
                explicit_risk_present = True
                rank = severity_rank[classification_risk]
                if rank > highest_rank:
                    highest_rank = rank
                    observed_risk = classification_risk

            if data.get("is_scam") is True or data.get("threat_detected") is True:
                explicit_threat_present = True

            # TrustPath contextual concern is a secondary signal. It does not
            # replace a module verdict, but it can prevent a multi-signal
            # suspicious claim from being treated as definitively clear when
            # no explicit threat classification exists.
            contextual = data.get("trustpath_contextual_signals")
            if isinstance(contextual, dict) and contextual.get("concern_present"):
                contextual_risk = contextual.get("risk_level", "UNKNOWN")
                contextual_rank = severity_rank.get(contextual_risk, 0)
                if contextual_rank > highest_rank:
                    highest_rank = contextual_rank
                    observed_risk = contextual_risk

            confidence = _safe_optional_float(data.get("confidence"))
            if confidence is not None:
                highest_confidence = max(highest_confidence, confidence)

        contextual_concern_present = any(
            isinstance(item.detected_data.get("trustpath_contextual_signals"), dict)
            and item.detected_data["trustpath_contextual_signals"].get("concern_present")
            for item in evidence
        )
        if contextual_concern_present and observed_risk != "UNKNOWN":
            explicit_risk_present = True

        # Biometric verification is explicit identity evidence, not an inferred
        # relationship. A labeled mismatch/impersonation result is treated as
        # HIGH risk; otherwise the explicit impersonation score determines the
        # conservative severity.
        biometric = _biometric_signals(evidence)
        label_upper = str(biometric["classification_label"] or "").upper()
        biometric_mismatch = biometric["match"] is False
        biometric_labeled_risk = (
            "IMPERSONATION" in label_upper
            or "MISMATCH" in label_upper
        )

        if biometric["present"] and (biometric_mismatch or biometric_labeled_risk):
            biometric_risk = "HIGH"
        elif biometric["present"] and biometric["impersonation_risk"] is not None:
            risk_score = biometric["impersonation_risk"]
            if risk_score >= 0.70:
                biometric_risk = "HIGH"
            elif risk_score >= 0.45:
                biometric_risk = "MEDIUM"
            else:
                biometric_risk = "LOW"
        else:
            biometric_risk = "UNKNOWN"

        if severity_rank[biometric_risk] > highest_rank:
            highest_rank = severity_rank[biometric_risk]
            observed_risk = biometric_risk
            explicit_risk_present = biometric_risk != "UNKNOWN"

        if explicit_risk_present:
            risk_level = observed_risk

        # If upstream only provides boolean threat verdicts, derive a
        # conservative severity from the highest explicit confidence.
        if risk_level == "UNKNOWN" and explicit_threat_present:
            if highest_confidence >= 0.90:
                risk_level = "CRITICAL"
            elif highest_confidence >= 0.75:
                risk_level = "HIGH"
            elif highest_confidence >= 0.50:
                risk_level = "MEDIUM"
            else:
                risk_level = "LOW"

    # Trust score is confidence in the case conclusion, not relationship
    # strength. A relationship can be very strong even when the shared claim is
    # suspicious, so it must never become the trust score by itself.
    if identity_confidence is not None:
        # Preserve the established biometric semantics: when explicit biometric
        # identity evidence exists, the score reflects identity confidence.
        trust_score = round(identity_confidence * 100, 2)
    elif scenario.confidence > 0.0 and scenario.scenario_type != "unknown":
        # Without biometric identity evidence, use confidence in the reconstructed
        # scenario rather than confidence in the evidence relationship graph.
        trust_score = round(scenario.confidence * 100, 2)
    else:
        trust_score = None

    return TrustPathAssessment(
        identity_confidence=identity_confidence,
        relationship_confidence=relationship_confidence,
        scenario_confidence=scenario.confidence,
        trust_score=trust_score,
        risk_level=risk_level,
    )


def _relationship_risk_level(
    relationship: TrustPathRelationship,
    contradictions: List[TrustPathContradiction],
) -> RiskLevel:
    """Map a relationship edge to stage risk without inheriting artifact risk."""
    if relationship.relation != "contradicts":
        return "LOW"

    matching = [
        contradiction.severity
        for contradiction in contradictions
        if set(contradiction.evidence_ids).issuperset(
            set(relationship.evidence_ids)
        )
        and contradiction.severity in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
    ]
    if not matching:
        return "HIGH"

    severity_rank = {
        "LOW": 1,
        "MEDIUM": 2,
        "HIGH": 3,
        "CRITICAL": 4,
    }
    return max(matching, key=lambda risk: severity_rank[risk])


def build_story(
    *,
    scenario: TrustPathScenario,
    evidence: List[EvidenceItem],
    relationships: List[TrustPathRelationship],
    events: List[TrustPathEvent],
    contradictions: List[TrustPathContradiction],
    assessment: TrustPathAssessment,
) -> TrustPathStory:
    """Create a human-readable story with risk attributed to its evidence."""

    evidence_by_id = {
        item.evidence_id: item
        for item in evidence
        if item.evidence_id
    }

    stage_list: List[TrustPathStage] = []

    for index, event in enumerate(events, start=1):
        event_evidence = [
            evidence_by_id[evidence_id]
            for evidence_id in event.evidence_ids
            if evidence_id in evidence_by_id
        ]
        event_risk = _risk_from_evidence_items(event_evidence)
        if event_risk == "UNKNOWN":
            event_risk = "LOW" if assessment.risk_level == "LOW" else "UNKNOWN"

        stage_list.append(
            TrustPathStage(
                stage_id=_new_id("stage"),
                sequence=index,
                title=event.title,
                summary=event.description,
                evidence_ids=event.evidence_ids,
                risk_impact=event_risk,
            )
        )

    for contradiction in contradictions:
        stage_list.append(
            TrustPathStage(
                stage_id=_new_id("stage"),
                sequence=len(stage_list) + 1,
                title=contradiction.title,
                summary=contradiction.description,
                evidence_ids=contradiction.evidence_ids,
                technique="Evidence inconsistency",
                risk_impact=contradiction.severity,
                protection=(
                    "Verify the conflicting information through "
                    "an independent trusted source."
                ),
            )
        )

    if relationships:
        relationship_summaries = [
            relationship.explanation
            or relationship.relation.replace("_", " ").capitalize()
            for relationship in relationships
        ]
        # Relationship stages describe the graph connection itself, not the
        # risk carried by the evidence participating in that connection. An
        # evidence item may be HIGH/CRITICAL while the relationship that links
        # it to another artifact is merely informational. Only an explicit
        # contradiction relationship carries contradiction severity here.
        contradiction_relationships = [
            relationship
            for relationship in relationships
            if relationship.relation == "contradicts"
        ]
        if contradiction_relationships:
            relationship_risk = max(
                (
                    _relationship_risk_level(
                        relationship,
                        contradictions,
                    )
                    for relationship in contradiction_relationships
                ),
                key=lambda risk: {
                    "UNKNOWN": 0,
                    "LOW": 1,
                    "MEDIUM": 2,
                    "HIGH": 3,
                    "CRITICAL": 4,
                }.get(risk, 0),
                default="UNKNOWN",
            )
        else:
            relationship_risk = "LOW"

        stage_list.append(
            TrustPathStage(
                stage_id=_new_id("stage"),
                sequence=len(stage_list) + 1,
                title="Cross-evidence relationship",
                summary=" ".join(relationship_summaries),
                evidence_ids=_deduplicate_strings(
                    [
                        evidence_id
                        for relationship in relationships
                        for evidence_id in relationship.evidence_ids
                    ]
                ),
                technique="Deterministic evidence correlation",
                risk_impact=relationship_risk,
            )
        )

    headline = scenario.title
    summary = scenario.summary
    current_manipulation = None
    victim_decision = None
    likely_next_harmful_action = None

    biometric = _biometric_signals(evidence)
    if biometric["present"] and biometric["match"] is True:
        current_manipulation = "No voice impersonation mismatch was detected."
        victim_decision = (
            "The supplied voice identity is consistent with the reference voice."
        )

    if biometric["present"] and biometric["match"] is False:
        current_manipulation = "A voice identity mismatch was detected."
        victim_decision = (
            "Do not rely on the supplied caller identity until it is "
            "independently verified."
        )

    for item in evidence:
        data = item.detected_data or {}
        if current_manipulation is None:
            current_manipulation = data.get("current_manipulation")
        if victim_decision is None:
            victim_decision = data.get("victim_decision")
        if likely_next_harmful_action is None:
            likely_next_harmful_action = data.get("likely_next_action")

    return TrustPathStory(
        headline=headline,
        summary=summary,
        current_manipulation=current_manipulation,
        victim_decision=victim_decision,
        likely_next_harmful_action=likely_next_harmful_action,
        stages=stage_list,
    )



# ---------------------------------------------------------------------------
# RECOMMENDATION
# ---------------------------------------------------------------------------

def build_recommendation(
    *,
    assessment: TrustPathAssessment,
    scenario: TrustPathScenario,
    contradictions: List[TrustPathContradiction],
) -> TrustPathRecommendation:
    """
    Generate a cautious, scenario-aware recommendation.

    High-risk cases remain protective. Verified biometric identity now receives
    a recommendation that reflects the successful verification instead of the
    generic "insufficient evidence" message.
    """

    if assessment.risk_level in {"HIGH", "CRITICAL"}:
        return TrustPathRecommendation(
            action=(
                "Pause the requested action and independently verify "
                "the sender, request, or transaction."
            ),
            reason=(
                "TrustPath identified high-severity inconsistencies "
                "or risk indicators."
            ),
            verification_required=True,
        )

    if contradictions:
        return TrustPathRecommendation(
            action=(
                "Do not rely on the conflicting information until it "
                "has been independently verified."
            ),
            reason=(
                "The evidence contains contradictions that reduce "
                "confidence in the current scenario."
            ),
            verification_required=True,
        )

    if scenario.scenario_type == "voice_identity_verified":
        return TrustPathRecommendation(
            action=(
                "The supplied voice matches the reference identity. "
                "Proceed with normal safeguards and use an independent "
                "channel before any high-impact action."
            ),
            reason=(
                "TrustPath found an explicit biometric voice match and "
                "no impersonation mismatch."
            ),
            verification_required=False,
        )

    if scenario.scenario_type == "voice_impersonation":
        return TrustPathRecommendation(
            action=(
                "Do not rely on the caller's voice identity. "
                "Independently verify the person through a trusted channel."
            ),
            reason=(
                "TrustPath found an explicit mismatch or impersonation "
                "signal in the biometric evidence."
            ),
            verification_required=True,
        )

    if scenario.confidence < 0.50:
        return TrustPathRecommendation(
            action=(
                "Gather additional evidence before making a "
                "high-impact decision."
            ),
            reason=(
                "The available evidence is not sufficient for a "
                "high-confidence TrustPath conclusion."
            ),
            verification_required=True,
        )

    if (
        assessment.risk_level == "LOW"
        and scenario.scenario_type != "unknown"
        and scenario.confidence >= 0.50
    ):
        return TrustPathRecommendation(
            action=(
                "The evidence supports the current TrustPath scenario. "
                "Continue with normal safeguards and independently verify "
                "any high-impact claim."
            ),
            reason=(
                "TrustPath reached a supported low-risk conclusion from "
                "explicit evidence."
            ),
            verification_required=False,
        )

    return TrustPathRecommendation(
        action=(
            "Review the evidence and independently verify important "
            "claims before taking a high-impact action."
        ),
        reason=(
            "TrustPath has not established enough evidence to justify "
            "a stronger conclusion."
        ),
        verification_required=True,
    )



# ---------------------------------------------------------------------------
# REPORT-READY RECORD
# ---------------------------------------------------------------------------

def build_report_metadata(
    case: TrustPathCase,
) -> Dict[str, Any]:
    """
    Prepare a structured report manifest.

    This does not generate a PDF yet.

    It gives the eventual report generator everything it needs to render:
        - case metadata
        - evidence
        - entities
        - relationships
        - scenario
        - stages
        - assessment
        - recommendation
    """

    return {
        "report_version": "1.0",
        "case_id": case.case_id,
        "generated_at": case.updated_at,
        "sections": [
            "case_overview",
            "input_sources",
            "evidence",
            "entities",
            "relationships",
            "events",
            "contradictions",
            "scenario",
            "trustpath_story",
            "assessment",
            "recommendation",
        ],
        "export_ready": True,
    }


# ---------------------------------------------------------------------------
# MAIN ORCHESTRATOR
# ---------------------------------------------------------------------------

def analyze_trustpath(payload: TrustPathInput) -> TrustPathCase:
    """
    Run the complete isolated TrustPath orchestration flow.

    Existing modules are not called or modified here.
    """

    now = _utc_now()

    evidence = normalize_evidence(payload)

    # Add TrustPath-only contextual observations without overwriting or
    # reinterpreting upstream module verdicts.
    _enrich_contextual_signals(evidence)

    # Add IDs to annotations.
    annotations: List[UserAnnotation] = []

    for annotation in payload.annotations:
        copied = annotation.model_copy(deep=True)

        if not copied.annotation_id:
            copied.annotation_id = _new_id("annotation")

        if not copied.created_at:
            copied.created_at = now

        annotations.append(copied)

    entities = extract_entities(evidence)

    relationships = build_relationships(
        entities=entities,
        evidence=evidence,
    )

    events = build_events(
        evidence=evidence,
    )

    contradictions = detect_contradictions(
        evidence=evidence,
        relationships=relationships,
    )

    scenario = reconstruct_scenario(
        evidence=evidence,
        events=events,
        contradictions=contradictions,
        relationships=relationships,
    )

    assessment = build_assessment(
        evidence=evidence,
        relationships=relationships,
        scenario=scenario,
        contradictions=contradictions,
    )

    story = build_story(
        scenario=scenario,
        evidence=evidence,
        relationships=relationships,
        events=events,
        contradictions=contradictions,
        assessment=assessment,
    )

    recommendation = build_recommendation(
        assessment=assessment,
        scenario=scenario,
        contradictions=contradictions,
    )

    input_sources = _deduplicate_strings(
        [
            item.source
            for item in evidence
            if item.source
        ]
    )

    case_id = _new_id("TP")

    case = TrustPathCase(
        case_id=case_id,
        created_at=now,
        updated_at=now,
        status="analyzed",
        case_name=payload.case_name,
        user_context=payload.user_context,
        input_sources=input_sources,
        evidence=evidence,
        annotations=annotations,
        entities=entities,
        relationships=relationships,
        events=events,
        contradictions=contradictions,
        scenario=scenario,
        assessment=assessment,
        story=story,
        recommendation=recommendation,
    )

    case.report = build_report_metadata(case)

    _CASE_STORE[case_id] = case

    return case


# ---------------------------------------------------------------------------
# TRUSTPATH INPUT ORCHESTRATION
# ---------------------------------------------------------------------------

TRUSTPATH_MODULE_STATUS = Literal[
    "NOT_PROVIDED",
    "INCOMPLETE",
    "RUNNING",
    "COMPLETED",
    "FAILED",
    "SKIPPED",
]


def _new_trustpath_target_id() -> str:
    """Create a temporary biometric enrollment target for this TrustPath run."""
    return f"trustpath_ref_{uuid4().hex[:12]}"


async def _trustpath_post_json(
    client: httpx.AsyncClient,
    path: str,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    """Call an existing JSON analysis endpoint through the same FastAPI app."""
    response = await client.post(path, json=payload)
    if response.status_code >= 400:
        raise RuntimeError(f"{path} returned HTTP {response.status_code}: {response.text[:500]}")
    data = response.json()
    if not isinstance(data, dict):
        raise RuntimeError(f"{path} returned a non-object response.")
    return data


async def _trustpath_post_file(
    client: httpx.AsyncClient,
    *,
    file: UploadFile,
    age_group: str,
    claimed_target_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Call the existing media analyzer without changing its implementation."""
    file_bytes = await file.read()
    if not file_bytes:
        raise RuntimeError(f"Empty file supplied: {file.filename or 'unnamed file'}")

    form_data = {"age_group": age_group}
    if claimed_target_id:
        form_data["claimed_target_id"] = claimed_target_id

    files = {
        "file": (
            file.filename or "trustpath-upload",
            file_bytes,
            file.content_type or "application/octet-stream",
        )
    }
    response = await client.post("/api/analyze/file", data=form_data, files=files)
    if response.status_code >= 400:
        raise RuntimeError(f"/api/analyze/file returned HTTP {response.status_code}: {response.text[:500]}")
    data = response.json()
    if not isinstance(data, dict):
        raise RuntimeError("/api/analyze/file returned a non-object response.")
    return data


async def _run_trustpath_text(
    client: httpx.AsyncClient,
    text: str,
    age_group: str,
) -> Dict[str, Any]:
    return await _trustpath_post_json(
        client,
        "/api/analyze/text",
        {"text": text, "age_group": age_group},
    )


async def _run_trustpath_file(
    client: httpx.AsyncClient,
    file_bytes: bytes,
    filename: str,
    content_type: Optional[str],
    age_group: str,
) -> Dict[str, Any]:
    """Analyze one media artifact using the existing /api/analyze/file route."""
    if not file_bytes:
        raise RuntimeError(f"Empty file supplied: {filename}")
    response = await client.post(
        "/api/analyze/file",
        data={"age_group": age_group},
        files={"file": (filename, file_bytes, content_type or "application/octet-stream")},
    )
    if response.status_code >= 400:
        raise RuntimeError(f"/api/analyze/file returned HTTP {response.status_code}: {response.text[:500]}")
    data = response.json()
    if not isinstance(data, dict):
        raise RuntimeError("/api/analyze/file returned a non-object response.")
    return data


async def _run_trustpath_biometric(
    client: httpx.AsyncClient,
    reference_file: UploadFile,
    test_file: UploadFile,
    age_group: str,
) -> Dict[str, Any]:
    """
    Run the existing biometric flow as one TrustPath branch:
    reference enrollment -> test verification.

    This branch is independent of Text/Visual/Acoustic and can execute in
    parallel with them.
    """
    reference_bytes = await reference_file.read()
    test_bytes = await test_file.read()
    if not reference_bytes or not test_bytes:
        raise RuntimeError("Reference Voice and Test Voice must both contain data.")

    target_id = _new_trustpath_target_id()

    enroll_response = await client.post(
        "/api/enroll/voice",
        data={"target_id": target_id},
        files={
            "file": (
                reference_file.filename or "trustpath-reference-voice",
                reference_bytes,
                reference_file.content_type or "application/octet-stream",
            )
        },
    )
    if enroll_response.status_code >= 400:
        raise RuntimeError(
            f"/api/enroll/voice returned HTTP {enroll_response.status_code}: {enroll_response.text[:500]}"
        )

    verify_response = await client.post(
        "/api/analyze/file",
        data={"age_group": age_group, "claimed_target_id": target_id},
        files={
            "file": (
                test_file.filename or "trustpath-test-voice",
                test_bytes,
                test_file.content_type or "application/octet-stream",
            )
        },
    )
    if verify_response.status_code >= 400:
        raise RuntimeError(
            f"/api/analyze/file (biometric) returned HTTP {verify_response.status_code}: "
            f"{verify_response.text[:500]}"
        )

    result = verify_response.json()
    if not isinstance(result, dict):
        raise RuntimeError("Biometric verification returned a non-object response.")
    result.setdefault("trustpath_reference_target_id", target_id)
    # Preserve the physical test artifact as TrustPath evidence metadata so
    # biometric verification can correlate with the Acoustic branch when the
    # same recording was supplied to both inputs.
    result.setdefault("source_reference", test_file.filename or "trustpath-test-voice")
    return result


async def _execute_trustpath_inputs(
    *,
    app: Any,
    text: Optional[str],
    visual: Optional[UploadFile],
    acoustic: Optional[UploadFile],
    reference_voice: Optional[UploadFile],
    test_voice: Optional[UploadFile],
    age_group: str,
    case_name: Optional[str],
    user_context: Optional[str],
) -> Dict[str, Any]:
    """Route only the evidence explicitly supplied to TrustPath."""
    manifest = TrustPathInputManifest(
        text=bool(text and text.strip()),
        visual=visual is not None,
        acoustic=acoustic is not None,
        reference_voice=reference_voice is not None,
        test_voice=test_voice is not None,
    )

    if not manifest.provided_sources():
        raise HTTPException(status_code=400, detail="TrustPath requires at least one evidence input.")

    module_status: Dict[str, str] = {
        "text": "READY" if manifest.text else "NOT_PROVIDED",
        "visual": "READY" if manifest.visual else "NOT_PROVIDED",
        "acoustic": "READY" if manifest.acoustic else "NOT_PROVIDED",
        "biometric": "READY" if manifest.biometric_ready() else (
            "INCOMPLETE" if manifest.reference_voice or manifest.test_voice else "NOT_PROVIDED"
        ),
    }

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://trustpath.local",
    ) as client:
        tasks: Dict[str, Any] = {}

        if manifest.text:
            tasks["text"] = asyncio.create_task(
                _run_trustpath_text(client, text.strip(), age_group)
            )

        # Read media into memory before parallel dispatch so each UploadFile is
        # consumed exactly once and the existing analyzer receives the original bytes.
        media_payloads: Dict[str, tuple[bytes, str, Optional[str]]] = {}
        for source, upload in (("visual", visual), ("acoustic", acoustic)):
            if upload is not None:
                raw = await upload.read()
                media_payloads[source] = (
                    raw,
                    upload.filename or f"trustpath-{source}",
                    upload.content_type,
                )

        for source, payload in media_payloads.items():
            module_status[source] = "RUNNING"
            raw, filename, content_type = payload
            tasks[source] = asyncio.create_task(
                _run_trustpath_file(client, raw, filename, content_type, age_group)
            )

        if manifest.biometric_ready():
            module_status["biometric"] = "RUNNING"
            tasks["biometric"] = asyncio.create_task(
                _run_trustpath_biometric(client, reference_voice, test_voice, age_group)
            )
        elif manifest.reference_voice or manifest.test_voice:
            module_status["biometric"] = "INCOMPLETE"

        module_results: Dict[str, Any] = {}
        module_errors: Dict[str, str] = {}

        if tasks:
            task_names = list(tasks.keys())
            results = await asyncio.gather(
                *(tasks[name] for name in task_names),
                return_exceptions=True,
            )
            for name, result in zip(task_names, results):
                if isinstance(result, Exception):
                    module_status[name] = "FAILED"
                    module_errors[name] = str(result)
                else:
                    module_status[name] = "COMPLETED"
                    module_results[name] = result

    case = analyze_trustpath(
        TrustPathInput(
            case_name=case_name or "LinguaShield TrustPath Investigation",
            user_context=user_context,
            evidence=[],
            annotations=[],
            existing_analysis=module_results,
        )
    )

    return {
        "case": case,
        "manifest": manifest,
        "module_status": module_status,
        "module_errors": module_errors,
    }


# ---------------------------------------------------------------------------
# API ENDPOINTS
# ---------------------------------------------------------------------------

@router.post(
    "/run",
)
async def run_trustpath_endpoint(
    request: Request,
    text: Optional[str] = Form(None),
    visual: Optional[UploadFile] = File(None),
    acoustic: Optional[UploadFile] = File(None),
    reference_voice: Optional[UploadFile] = File(None),
    test_voice: Optional[UploadFile] = File(None),
    age_group: str = Form("adult"),
    case_name: Optional[str] = Form(None),
    user_context: Optional[str] = Form(None),
) -> Dict[str, Any]:
    """
    Run a self-contained TrustPath investigation from its own evidence inputs.

    Only supplied evidence branches are executed. Existing standalone module
    routes are reused through the same FastAPI application; those routes are
    not modified by this orchestrator.
    """
    return await _execute_trustpath_inputs(
        app=request.app,
        text=text,
        visual=visual,
        acoustic=acoustic,
        reference_voice=reference_voice,
        test_voice=test_voice,
        age_group=age_group,
        case_name=case_name,
        user_context=user_context,
    )


@router.post(
    "/analyze",
    response_model=TrustPathCase,
)
async def analyze_endpoint(
    payload: TrustPathInput,
) -> TrustPathCase:
    """
    Analyze a TrustPath case.

    This endpoint is intentionally separate from the existing analysis routes.
    """

    try:
        return analyze_trustpath(payload)

    except Exception as exc:
        # Avoid leaking implementation details to the client.
        raise HTTPException(
            status_code=500,
            detail="TrustPath analysis failed.",
        ) from exc


@router.post(
    "/analyze/modules",
    response_model=TrustPathCase,
)
async def analyze_modules_endpoint(
    bundle: TrustPathModuleBundle,
) -> TrustPathCase:
    """
    Analyze a TrustPath case from existing Text/Visual/Acoustic/Biometric
    module outputs using the shared adapter contract.

    The existing module outputs are treated as read-only inputs.
    """
    payload = TrustPathInput(
        case_name=bundle.case_name,
        user_context=bundle.user_context,
        evidence=bundle.evidence,
        annotations=bundle.annotations,
        existing_analysis={
            "text": bundle.text or None,
            "visual": bundle.visual or None,
            "acoustic": bundle.acoustic or None,
            "biometric": bundle.biometric or None,
        },
    )

    return analyze_trustpath(payload)


@router.get(
    "/case/{case_id}",
    response_model=TrustPathCase,
)
async def get_case(
    case_id: str,
) -> TrustPathCase:
    """Retrieve a previously analyzed TrustPath case."""

    case = _CASE_STORE.get(case_id)

    if case is None:
        raise HTTPException(
            status_code=404,
            detail="TrustPath case not found.",
        )

    return deepcopy(case)


@router.get(
    "/case/{case_id}/report",
)
async def get_report_record(
    case_id: str,
) -> Dict[str, Any]:
    """
    Return the structured report-ready case record.

    A future PDF/HTML/JSON exporter can consume this response directly.
    """

    case = _CASE_STORE.get(case_id)

    if case is None:
        raise HTTPException(
            status_code=404,
            detail="TrustPath case not found.",
        )

    return {
        "case": case.model_dump(mode="json"),
        "report": deepcopy(case.report),
    }


@router.get(
    "/health",
)
async def trustpath_health() -> Dict[str, Any]:
    """
    Lightweight health endpoint for isolated TrustPath testing.
    """

    return {
        "service": "trustpath",
        "status": "ok",
        "cases_in_memory": len(_CASE_STORE),
    }


# ---------------------------------------------------------------------------
# OPTIONAL DEVELOPMENT HELPERS
# ---------------------------------------------------------------------------

def clear_in_memory_cases() -> None:
    """
    Development/testing helper.

    Not exposed as a public API route intentionally.
    """

    _CASE_STORE.clear()


__all__ = [
    "router",
    "TrustPathInput",
    "TrustPathInputManifest",
    "TrustPathModuleBundle",
    "EvidenceItem",
    "UserAnnotation",
    "TrustPathCase",
    "TrustPathEntity",
    "TrustPathRelationship",
    "TrustPathEvent",
    "TrustPathContradiction",
    "TrustPathScenario",
    "TrustPathAssessment",
    "TrustPathStage",
    "TrustPathStory",
    "TrustPathRecommendation",
    "analyze_trustpath",
    "normalize_existing_analysis",
    "clear_in_memory_cases",
]


class TrustPathInput(BaseModel):
    """
    Payload submitted to TrustPath.

    existing_analysis contains the output already generated by the existing
    LinguaShield modules. TrustPath does not re-run those modules.

    The optional `inputs` manifest describes which TrustPath evidence types
    are present for a self-contained investigation. It is intentionally a
    data-contract field in this phase; execution/routing is added separately
    without changing the existing module endpoints.
    """

    case_name: Optional[str] = None

    user_context: Optional[str] = Field(
        default=None,
        description="Optional overall context supplied by the user."
    )

    evidence: List[EvidenceItem] = Field(default_factory=list)

    annotations: List[UserAnnotation] = Field(default_factory=list)

    existing_analysis: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Existing Text/Visual/Acoustic/Biometric analysis output. "
            "TrustPath treats this as read-only input."
        ),
    )

    inputs: "TrustPathInputManifest" = Field(
        default_factory=lambda: TrustPathInputManifest(),
        description=(
            "Evidence-presence manifest for the TrustPath workspace. "
            "It does not execute analyzers in this phase."
        ),
    )