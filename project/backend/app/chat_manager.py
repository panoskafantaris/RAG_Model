"""
Enhanced chat manager with proper error handling and data validation.
For production: Replace with database (PostgreSQL/MongoDB).
"""
import uuid
from datetime import datetime
from typing import Dict, List, Optional
from threading import Lock

from .exceptions import ChatNotFoundException
from .logger import setup_logger

logger = setup_logger(__name__)

# In-memory storage
CHATS: Dict[str, Dict] = {}

# Thread-safe operations
_lock = Lock()


def create_chat(title: str = "Νέα Συνομιλία") -> str:
    """
    Create a new chat session.
    
    Args:
        title: Chat title
    
    Returns:
        Chat ID (UUID)
    """
    chat_id = str(uuid.uuid4())
    
    now = datetime.utcnow().isoformat()
    
    with _lock:
        CHATS[chat_id] = {
            "id": chat_id,
            "title": title,
            "messages": [],
            "created": now,
            "updated": now,
        }
    
    logger.info(f"Created new chat: {chat_id} - '{title}'")
    return chat_id


def list_chats() -> List[Dict]:
    """
    List all chats sorted by last update (newest first).
    
    Returns:
        List of chat summaries
    """
    with _lock:
        chats = [
            {
                "id": c["id"],
                "title": c["title"],
                "last_updated": c["updated"],
                "message_count": len(c["messages"])
            }
            for c in CHATS.values()
        ]
    
    # Sort by last updated (newest first)
    chats.sort(key=lambda x: x["last_updated"], reverse=True)
    
    return chats


def get_chat(chat_id: str) -> Dict:
    """
    Get full chat details.
    
    Args:
        chat_id: Chat ID
    
    Returns:
        Chat dictionary
    
    Raises:
        ChatNotFoundException: If chat doesn't exist
    """
    with _lock:
        if chat_id not in CHATS:
            raise ChatNotFoundException(chat_id)
        return CHATS[chat_id].copy()


def append_message(chat_id: str, role: str, content: str) -> Dict:
    """
    Append a message to chat history.
    
    Args:
        chat_id: Chat ID
        role: Message role ("user" or "assistant")
        content: Message content
    
    Returns:
        The created message entry
    
    Raises:
        ChatNotFoundException: If chat doesn't exist
    """
    if chat_id not in CHATS:
        raise ChatNotFoundException(chat_id)
    
    now = datetime.utcnow().isoformat()
    
    message = {
        "role": role,
        "content": content,
        "timestamp": now
    }
    
    with _lock:
        CHATS[chat_id]["messages"].append(message)
        CHATS[chat_id]["updated"] = now
    
    logger.debug(f"Appended {role} message to chat {chat_id}")
    
    return message


def get_history(chat_id: str) -> List[Dict]:
    """
    Get chat message history.
    
    Args:
        chat_id: Chat ID
    
    Returns:
        List of messages
    
    Raises:
        ChatNotFoundException: If chat doesn't exist
    """
    if chat_id not in CHATS:
        raise ChatNotFoundException(chat_id)
    
    with _lock:
        return CHATS[chat_id]["messages"].copy()


def update_chat_title(chat_id: str, title: str) -> None:
    """
    Update chat title.
    
    Args:
        chat_id: Chat ID
        title: New title
    
    Raises:
        ChatNotFoundException: If chat doesn't exist
    """
    if chat_id not in CHATS:
        raise ChatNotFoundException(chat_id)
    
    with _lock:
        CHATS[chat_id]["title"] = title
        CHATS[chat_id]["updated"] = datetime.utcnow().isoformat()
    
    logger.info(f"Updated title for chat {chat_id}: '{title}'")


def delete_chat(chat_id: str) -> None:
    """
    Delete a chat.
    
    Args:
        chat_id: Chat ID
    
    Raises:
        ChatNotFoundException: If chat doesn't exist
    """
    if chat_id not in CHATS:
        raise ChatNotFoundException(chat_id)
    
    with _lock:
        del CHATS[chat_id]
    
    logger.info(f"Deleted chat {chat_id}")


def clear_all_chats() -> int:
    """
    Clear all chats (useful for testing).
    
    Returns:
        Number of chats deleted
    """
    with _lock:
        count = len(CHATS)
        CHATS.clear()
    
    logger.info(f"Cleared {count} chats")
    return count


def get_stats() -> Dict:
    """
    Get system statistics.
    
    Returns:
        Dictionary with stats
    """
    with _lock:
        total_chats = len(CHATS)
        total_messages = sum(len(c["messages"]) for c in CHATS.values())
    
    return {
        "total_chats": total_chats,
        "total_messages": total_messages,
        "avg_messages_per_chat": total_messages / total_chats if total_chats > 0 else 0
    }