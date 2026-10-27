from services.trustpath.orchestrator import (
    run_trustpath,
    validate_orchestration_result,
)


# ============================================================
# TRUSTPATH TEST INPUT
# ============================================================

analysis = {
    "text": {
        "classification": "suspicious",
        "confidence": 0.91,
        "text": (
            "Rahul says he is my manager and asks me "
            "to make an urgent payment."
        ),
        "intent": "financial_request",

        # Explicit structured field for TrustPath entity extraction
        "person_name": "Rahul",

        # Explicit event information for the current event layer
        "event_title": "Urgent Payment Request",
        "event_description": (
            "Rahul requested that the user make an urgent payment."
        ),
    },

    "visual": {
        "classification": "payment_document",
        "confidence": 0.84,

        # Explicit visual entity information
        "document_name": "Payment Document",
        "invoice_number": "INV-2026-001",
    },

    "acoustic": {
        "vocoder_score": 0.88,
        "synthetic_detected": True,
        "confidence": 0.90,

        # Explicit speaker/entity information
        "speaker_name": "Rahul",
    },

    "biometric": {
        "match": False,
        "confidence": 0.20,
    },
}


# ============================================================
# RUN TRUSTPATH
# ============================================================

result = run_trustpath(
    analysis,
    user_context="Rahul, my manager",
)


# ============================================================
# SMALL OUTPUT COMPATIBILITY HELPERS
# ============================================================

def get_collection(value, key):
    """
    Return a collection from TrustPath output.

    Current TrustPath modules may return:
        - a list directly
        - a dictionary containing the collection
        - None

    This helper only normalizes the test output.
    It does not modify the TrustPath pipeline.
    """

    if isinstance(value, list):
        return value

    if isinstance(value, dict):
        collection = value.get(key)

        if isinstance(collection, list):
            return collection

        if collection is None:
            return []

    return []


def get_object(value, fallback=None):
    """
    Return a dictionary object when available.
    """

    if isinstance(value, dict):
        return value

    if fallback is not None:
        return fallback

    return {}


# ============================================================
# HEADER
# ============================================================

print("========================================")
print("TRUSTPATH ORCHESTRATOR TEST")
print("========================================")


# ============================================================
# EVIDENCE
# ============================================================

print("\nEVIDENCE:")

evidence = get_collection(
    result.get("evidence", []),
    "evidence",
)

print(len(evidence))

for item in evidence:
    print(
        item.get("evidence_id"),
        "|",
        item.get("source"),
        "|",
        item.get("title"),
        "|",
        item.get("status"),
    )


# ============================================================
# ENTITIES
# ============================================================

print("\nENTITIES:")

entities = get_collection(
    result.get("entities", []),
    "entities",
)

print(len(entities))

for entity in entities:
    print(
        entity.get("entity_id"),
        "|",
        entity.get(
            "entity_type",
            entity.get("type"),
        ),
        "|",
        entity.get(
            "name",
            entity.get(
                "canonical_name",
                entity.get("value"),
            ),
        ),
        "|",
        entity.get(
            "source_evidence_ids",
            entity.get("evidence_ids", []),
        ),
    )


# ============================================================
# RELATIONSHIPS
# ============================================================

print("\nRELATIONSHIPS:")

relationships = get_collection(
    result.get("relationships", []),
    "relationships",
)

print(len(relationships))

for relationship in relationships:
    print(
        relationship.get("relationship_id"),
        "|",
        relationship.get(
            "relation",
            relationship.get("relationship_type"),
        ),
        "|",
        relationship.get(
            "subject_entity_id",
            relationship.get("source_entity_id"),
        ),
        "->",
        relationship.get(
            "object_entity_id",
            relationship.get("target_entity_id"),
        ),
        "|",
        relationship.get("status"),
        "|",
        relationship.get("evidence_ids", []),
    )


# ============================================================
# EVENTS
# ============================================================

print("\nEVENTS:")

events = get_collection(
    result.get("events", []),
    "events",
)

print(len(events))

for event in events:
    print(
        event.get("event_id"),
        "|",
        event.get(
            "event_type",
            event.get("type"),
        ),
        "|",
        event.get("title"),
        "|",
        event.get("description"),
        "|",
        event.get("evidence_ids", []),
    )


# ============================================================
# SCENARIO
# ============================================================

print("\nSCENARIO:")

scenario_result = result.get(
    "scenario",
    {},
)

if isinstance(
    scenario_result,
    dict,
):
    scenario = scenario_result.get(
        "scenario",
        scenario_result,
    )
else:
    scenario = {}


print(
    scenario.get(
        "scenario_type",
        "unknown",
    )
)

print(
    "CONFIDENCE:",
    scenario.get(
        "confidence",
        0.0,
    ),
)


# ============================================================
# CONTRADICTIONS
# ============================================================

print("\nCONTRADICTIONS:")

contradictions = get_collection(
    result.get("contradictions", []),
    "contradictions",
)

print(len(contradictions))

for contradiction in contradictions:
    print(
        contradiction.get(
            "contradiction_id",
        ),
        "|",
        contradiction.get(
            "contradiction_type",
        ),
        "|",
        contradiction.get(
            "severity",
        ),
        "|",
        contradiction.get(
            "evidence_ids",
            [],
        ),
    )


# ============================================================
# ASSESSMENT
# ============================================================

print("\nASSESSMENT:")

assessment = result.get(
    "assessment",
    {},
)

print(assessment)


# ============================================================
# RECOMMENDATION
# ============================================================

print("\nRECOMMENDATION:")

recommendation = result.get(
    "recommendation",
    {},
)

print(recommendation)


# ============================================================
# STORY
# ============================================================

print("\nSTORY:")

story_result = result.get(
    "story",
    {},
)

if isinstance(
    story_result,
    dict,
):
    story = story_result

    print(
        story.get(
            "headline",
            "",
        )
    )

    print(
        story.get(
            "summary",
            "",
        )
    )

else:
    story = {}


# ============================================================
# STORY STAGES
# ============================================================

print("\nSTAGES:")

story_stages = story.get(
    "stages",
    [],
)

if isinstance(
    story_stages,
    list,
):

    for stage in story_stages:

        print(
            stage.get(
                "stage_type",
                stage.get(
                    "stage_id",
                    stage.get(
                        "stage",
                        "unknown",
                    ),
                ),
            ),
            "|",
            stage.get(
                "title",
                "",
            ),
            "|",
            stage.get(
                "evidence_ids",
                [],
            ),
        )


# ============================================================
# TRACEABILITY
# ============================================================

print("\nTRACEABILITY:")

traceability = result.get(
    "traceability",
    {},
)

print(traceability)


# ============================================================
# VALIDATION
# ============================================================

print("\nVALIDATION:")

validation = validate_orchestration_result(
    result
)

print(validation)


# ============================================================
# EXTRA DEBUG INFORMATION
# ============================================================

print("\n========================================")
print("TRUSTPATH DEBUG")
print("========================================")


# ============================================================
# INPUT SOURCES
# ============================================================

print("\nINPUT SOURCES:")

input_sources = result.get(
    "input_sources",
    [],
)

if not input_sources:
    input_sources = [
        item.get("source")
        for item in evidence
        if isinstance(item, dict) and item.get("source")
    ]

print(input_sources)


# ============================================================
# USER CONTEXT
# ============================================================

print("\nUSER CONTEXT:")

user_context_result = result.get(
    "user_context",
    None,
)

if user_context_result is None:
    user_context_result = (
        result.get("normalized", {}).get("user_context")
        if isinstance(result.get("normalized"), dict)
        else None
    )

if user_context_result is None:
    user_context_result = "Rahul, my manager"

print(user_context_result)


# ============================================================
# ENTITY SUMMARY
# ============================================================

print("\nENTITY SUMMARY:")

entity_summary = result.get(
    "entity_summary",
    result.get(
        "entities_summary",
        {},
    ),
)

if not entity_summary:
    entities_result = result.get(
        "entity_result",
        {},
    )

    if isinstance(entities_result, dict):
        entity_summary = entities_result.get(
            "summary",
            {},
        )

if not entity_summary:
    entity_types = {}

    for entity in entities:
        entity_type = (
            entity.get("entity_type")
            or entity.get("type")
            or "unknown"
        )

        entity_types[entity_type] = (
            entity_types.get(entity_type, 0) + 1
        )

    entity_summary = {
        "total": len(entities),
        "by_type": entity_types,
    }

print(entity_summary)


# ============================================================
# RELATIONSHIP SUMMARY
# ============================================================

print("\nRELATIONSHIP SUMMARY:")

relationship_summary = result.get(
    "relationship_summary",
    result.get(
        "relationships_summary",
        {},
    ),
)

if not relationship_summary:
    relationships_result = result.get(
        "relationship_result",
        {},
    )

    if isinstance(relationships_result, dict):
        relationship_summary = relationships_result.get(
            "summary",
            {},
        )

if not relationship_summary:
    by_type = {}
    by_status = {}

    for relationship in relationships:
        relationship_type = (
            relationship.get("relationship_type")
            or relationship.get("relation")
            or "unknown"
        )

        status = (
            relationship.get("status")
            or "unknown"
        )

        by_type[relationship_type] = (
            by_type.get(relationship_type, 0) + 1
        )

        by_status[status] = (
            by_status.get(status, 0) + 1
        )

    relationship_summary = {
        "total": len(relationships),
        "by_type": by_type,
        "by_status": by_status,
    }

print(relationship_summary)


# ============================================================
# EVENT SUMMARY
# ============================================================

print("\nEVENT SUMMARY:")

event_summary = result.get(
    "event_summary",
    result.get(
        "events_summary",
        {},
    ),
)

if not event_summary:
    event_types = {}

    for event in events:
        event_type = (
            event.get("event_type")
            or event.get("type")
            or "unknown"
        )

        event_types[event_type] = (
            event_types.get(event_type, 0) + 1
        )

    event_summary = {
        "total": len(events),
        "by_type": event_types,
    }

print(event_summary)


# ============================================================
# FINAL CHECK
# ============================================================

print("\n========================================")
print("TRUSTPATH TEST COMPLETE")
print("========================================")