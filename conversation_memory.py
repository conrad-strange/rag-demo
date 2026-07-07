from typing import Dict, List, Optional

from db import list_recent_chat_messages


def load_conversation_memory(
    session_id: Optional[str],
    enabled: bool = True,
    turns: int = 4,
) -> Dict:
    if not enabled or not session_id:
        return {"enabled": False, "history_count": 0, "history": [], "prefix": ""}

    limit = max(1, min(int(turns), 10))
    history = list_recent_chat_messages(session_id, limit=limit)
    if not history:
        return {"enabled": True, "history_count": 0, "history": [], "prefix": ""}

    lines: List[str] = [
        "Conversation history for resolving follow-up references only.",
        "Do not treat history as retrieved evidence; factual claims still need retrieved sources.",
    ]
    for index, item in enumerate(history, start=1):
        query = (item.get("query") or "").strip()
        answer = (item.get("answer") or "").strip()
        if len(answer) > 700:
            answer = answer[:700].rstrip() + "..."
        lines.append(f"Turn {index} user: {query}")
        lines.append(f"Turn {index} assistant: {answer}")

    return {
        "enabled": True,
        "history_count": len(history),
        "history": history,
        "prefix": "\n".join(lines),
    }


def apply_memory_to_query(query: str, memory: Dict) -> str:
    prefix = memory.get("prefix")
    if not prefix:
        return query
    return f"{prefix}\n\nCurrent user question:\n{query}"
