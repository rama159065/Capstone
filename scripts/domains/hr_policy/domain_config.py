from scripts.agents.rag_agent import DomainConfig

DOMAIN_CONFIG = DomainConfig(
    domain="hr_policy",
    display_name="HR Policy Navigator",
    description="Cited RAG navigator for leave, telework, wage-hour, worker rights, and workplace safety policy.",
    intent_terms=(
        "hr", "employee", "employer", "leave", "fmla", "telework", "remote work",
        "wage", "hour", "worker rights", "safety", "policy",
    ),
    review_terms=(
        "terminate", "discipline", "fire", "hire", "accommodation", "legal advice",
        "personal data", "employee record", "retaliation",
    ),
    source_types=(
        "hr_leave_policy_pdf", "hr_telework_policy_pdf",
        "hr_wage_hour_policy_pdf", "hr_safety_policy_pdf",
    ),
    default_query="What does the FMLA employee guide say about requesting leave?",
)
