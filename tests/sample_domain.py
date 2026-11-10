"""A DomainConfig for testing the shared domain-agent code (scripts/agents/rag_agent.py).

The product serves financial report QA only; the per-domain packages were removed. rag_agent.py stays
(the multimodal routes use its input guard), so its graph, guards and filters are tested with this config.
"""
from scripts.agents.rag_agent import DomainConfig

SAMPLE_DOMAIN = DomainConfig(
    domain="procurement_vendor",
    display_name="Procurement and Vendor-Review Assistant",
    description="Cited RAG assistant for procurement policy, vendor evaluation, and bid-review guidance.",
    intent_terms=(
        "procurement", "vendor", "supplier", "bid", "proposal", "contract", "award",
        "far", "rated criteria", "borrower", "qualification", "evaluation",
    ),
    review_terms=(
        "award", "disqualify", "approve vendor", "reject vendor", "fraud",
        "conflict of interest", "debar", "sanction",
    ),
    source_types=(
        "procurement_policy_pdf", "procurement_guidance_pdf",
        "procurement_directive_pdf", "vendor_evaluation_guidance_pdf",
        "federal_procurement_regulation_pdf",
    ),
    default_query="What evidence describes vendor evaluation criteria in procurement guidance?",
)
