from scripts.domain_common.rag_agent import DomainConfig

DOMAIN_CONFIG = DomainConfig(
    domain="competitive_research",
    display_name="Competitive-Research Analyst",
    description="Cited RAG analyst for competitor annual reports, risk language, market positioning, and comparison briefs.",
    intent_terms=(
        "competitive", "competitor", "annual report", "risk", "market", "revenue",
        "product", "business", "apple", "microsoft", "amazon", "nvidia", "compare",
    ),
    review_terms=(
        "buy", "sell", "hold", "price target", "investment advice", "recommend",
        "overvalued", "undervalued", "publish", "investor memo",
    ),
    source_types=("competitor_annual_report_pdf",),
    default_query="Compare competitor risk language about revenue and market demand.",
)
