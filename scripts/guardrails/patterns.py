"""Regex sets for the deterministic guardrails.

Secrets and sensitive-data patterns are shared. The financial and domain agents currently use
slightly different prompt-injection and code/tool sets, kept side by side here so the difference
is explicit (unifying them would change what one agent blocks):

- financial adds "act as developer/system/admin" and a broader SQL rule (any of
  select/insert/update/delete/drop ... from/table/database) plus closing ``</script>``;
- domain uses four narrower SQL rules.
"""

FINANCIAL_PROMPT_INJECTION_PATTERNS = [
    r"\bignore (all )?(previous|prior|above) instructions\b",
    r"\breveal (the )?(system|developer) prompt\b",
    r"\bact as (a )?(developer|system|admin)\b",
    r"\boverride (the )?(policy|instructions|guardrails)\b",
    r"\bdisregard (the )?(policy|instructions)\b",
]
DOMAIN_PROMPT_INJECTION_PATTERNS = [
    r"\bignore (all )?(previous|prior|above) instructions\b",
    r"\breveal (the )?(system|developer) prompt\b",
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
DOMAIN_CODE_OR_TOOL_PATTERNS = [
    r"<\s*script\b",
    r"```",
    r"\bselect\s+.+\bfrom\b",
    r"\bdrop\s+table\b",
    r"\binsert\s+into\b",
    r"\bdelete\s+from\b",
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

INPUT_BLOCKING_LABELS = {"prompt_injection", "code_or_tool_payload", "secret_or_credential"}
OUTPUT_BLOCKING_LABELS = {"prompt_injection_echo", "code_or_tool_payload", "secret_or_credential"}
