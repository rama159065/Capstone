from scripts.agents.rag_agent import DomainConfig

DOMAIN_CONFIG = DomainConfig(
    domain="incident_postmortem",
    display_name="Incident Postmortem Drafter",
    description="Cited RAG drafter for incident response, recovery, forensics, timelines, and postmortem action items.",
    intent_terms=(
        "incident", "postmortem", "outage", "response", "recovery", "containment",
        "eradication", "forensics", "timeline", "root cause", "nist",
    ),
    review_terms=(
        "publish", "public postmortem", "customer impact", "root cause", "blame",
        "credential", "exploit", "vulnerability",
    ),
    source_types=(
        "incident_response_guidance_pdf", "incident_recovery_guidance_pdf",
        "incident_forensics_guidance_pdf",
    ),
    default_query="What should be included in an incident postmortem according to incident response guidance?",
)
