from scripts.agents.rag_agent import DomainConfig

DOMAIN_CONFIG = DomainConfig(
    domain="travel_expense",
    display_name="Travel-and-Expense Policy Agent",
    description="Cited RAG assistant for travel reimbursement, documentation, mileage, meals, and expense policy.",
    intent_terms=(
        "travel", "expense", "reimbursement", "per diem", "mileage", "meals",
        "lodging", "car", "gift", "receipt", "jtr", "irs publication 463",
    ),
    review_terms=(
        "approve reimbursement", "deny reimbursement", "exception", "pay", "payment",
        "high value", "card number", "personal card",
    ),
    source_types=("tax_travel_expense_pdf", "travel_policy_pdf"),
    default_query="What documentation is required for travel expenses?",
)
