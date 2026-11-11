import re


def is_unsafe_retrieval_query(query: str) -> bool:
    value = query.strip()
    patterns = (
        r"^(?:https?://|www\.)",
        r"^(?:select|insert|update|delete|drop|alter|create)\b",
        r"^(?:sudo|curl|wget|bash|sh|zsh|powershell|cmd|rm|chmod)\b",
        r"^(?:调用|使用).{0,20}(?:工具|tool)",
        r"^use\s+.{0,30}\s+tool\b",
    )
    return any(re.search(pattern, value, flags=re.IGNORECASE) for pattern in patterns)
