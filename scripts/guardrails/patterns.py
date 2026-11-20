"""Regex sets for the deterministic guardrails (financial report copilot).

The domain-agent pattern sets were removed with scripts/agents/rag_agent.py on 11 Nov 2026.
"ignore your/the previous instructions" was missed until a notebook 05 run found it (eval item FIN-082).
"""

FINANCIAL_PROMPT_INJECTION_PATTERNS = [
    r"\bignore (all |any |your |the |my )?(previous|prior|above|earlier) (instructions|rules)\b",
    r"\breveal (the )?(system|developer) prompt\b",
    r"\bact as (a )?(developer|system|admin)\b",
    r"\boverride (the )?(policy|instructions|guardrails)\b",
    r"\bdisregard (the )?(policy|instructions)\b",
]
FINANCIAL_CODE_OR_TOOL_PATTERNS = [
    r"<\s*script\b",
    r"</\s*script\s*>",
    r"```",
    r"\b(select|insert|update|delete|drop)\s+.+\b(from|table|database)\b",
    r"\b(rm\s+-rf|curl\s+|wget\s+|powershell\s+|cmd\.exe|bash\s+)\b",
    r"\btool_call\b|\bfunction_call\b",
]
SECRET_PATTERNS = [
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
    r"\bsk-[A-Za-z0-9_-]{12,}\b",
    r"\b[A-Za-z0-9_]*(api|secret|token|password)[A-Za-z0-9_]*\s*[:=]\s*[^\s]{8,}",
]
SENSITIVE_DATA_PATTERNS = [
    r"\b(?:\d[ -]*?){13,19}\b",
    r"\b\d{3}-\d{2}-\d{4}\b",
    r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
]
BASE64_LIKE_PATTERN = r"\b[A-Za-z0-9+/]{120,}={0,2}\b"

# Requests the copilot refuses by policy, decided before any LLM call (Week 1 section 4.2: a small model given the
# refusal rule in its prompt still gave a buy recommendation, and a decomposition template answered advice requests).
# Narrow on purpose: a question about what a filing *reports* (risk factors, guidance, outlook) is not matched.
FINANCIAL_POLICY_REFUSALS = {
    "investment_advice": [
        r"\bshould (i|we) (buy|sell|hold|short|invest)\b",
        r"\b(buy|sell),? (or|vs\.?) (sell|hold|buy)\b",
        r"\bis \S+(\s\S+){0,4} a (good|bad) (buy|investment|stock)\b",
        r"\b(safest|riskiest)\b.{0,80}\b(keep|put|invest|park) (my|our) (money|savings|cash)\b",
        r"\bwhere (to|should i) (keep|put|invest) (my|our) (money|savings|cash)\b",
        r"\b(recommend|suggest) (a |which )?(stock|stocks|shares|investment)s?\b",
    ],
    "forecast_or_prediction": [
        r"\bpredict\b",
        r"\bforecast (the|its|their|what|how|next)\b",
        r"\bwhat will\b.{0,80}\b(be|reach|look like)\b",
        r"\bwill .{0,60}\b(beat|miss|exceed)\b.{0,40}\b(expectations|estimates|consensus)\b",
        r"\b(share|stock) price (at the end of|by the end of|next|in \d{4})\b",
        r"\bprice target\b",
    ],
    "personal_data": [
        r"\b(home|personal|residential|private) (address|phone( number)?|e-?mail|cell)\b",
    ],
    "non_public_information": [
        r"\bnon[- ]public\b",
        r"\binsider (information|tips?|knowledge)\b",
        r"\b(leaked|unreleased|upcoming) (quarterly |annual )?(results|earnings)\b",
    ],
    "off_topic_creative": [
        r"\b(write|compose|make up) (me )?(a |an )?(poem|song|joke|story|limerick|rap|haiku)\b",
    ],
}
POLICY_REFUSAL_MESSAGES = {
    "investment_advice": ("I can't give investment advice or recommendations. I can tell you what the companies' "
                          "annual reports state, with citations: for example revenue, net income, cash flow or debt."),
    "forecast_or_prediction": ("I can't predict future results, prices or earnings. The annual reports cover past "
                               "fiscal years; ask for a reported figure, or for what a company itself says about "
                               "its outlook."),
    "personal_data": "I can't share personal information about individuals.",
    "non_public_information": ("I only use published annual reports, so I can't provide non-public or inside "
                               "information."),
    "off_topic_creative": ("This copilot answers questions about companies' annual reports; it doesn't write "
                           "creative pieces."),
}

INPUT_BLOCKING_LABELS = {"prompt_injection", "code_or_tool_payload", "secret_or_credential"}
OUTPUT_BLOCKING_LABELS = {"prompt_injection_echo", "code_or_tool_payload", "secret_or_credential"}
