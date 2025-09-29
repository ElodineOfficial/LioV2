# chat_store.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
from datetime import datetime

from chatter import ChatMessage


@dataclass
class Entry:
    role: str            # 'user' or 'assistant'
    author_id: int       # 0 for assistant/system
    author_name: str     # assistant/system can be 'Lio'
    content: str
    created_at: datetime


class ConversationBuffer:
    """Rolling transcript with hard char cap; purges oldest→newest when over limit."""
    def __init__(self, limit_chars: int = 10_000):
        self.limit_chars = limit_chars
        self.entries: List[Entry] = []

    def _total_chars(self) -> int:
        return sum(len(e.content) for e in self.entries)

    def add(self, entry: Entry):
        self.entries.append(entry)
        self._enforce_limit()

    def _enforce_limit(self):
        while self._total_chars() > self.limit_chars and self.entries:
            self.entries.pop(0)

    def export(self, include_author_names: bool = True) -> List[ChatMessage]:
        """
        Convert internal entries → provider-agnostic ChatMessage[].
        If include_author_names=True, prefix user lines with 'Name: ' for grounding.
        """
        msgs: List[ChatMessage] = []
        for e in self.entries:
            if e.role == "assistant":
                msgs.append(ChatMessage(role="assistant", content=e.content))
            else:
                content = e.content
                if include_author_names and e.author_name:
                    content = f"{e.author_name}: {content}"
                msgs.append(ChatMessage(role="user", content=content))
        return msgs


class ConversationStore:
    """
    Channel-scoped faux thread:
      - One rolling buffer per channel
      - Record: user messages (including @bot), and assistant messages
      - 10,000-char (configurable) global limit per channel buffer
    """
    def __init__(self, limit_chars: int = 10_000):
        self.limit_chars = limit_chars
        self.buffers: Dict[int, ConversationBuffer] = {}

    def _buf(self, channel_id: int) -> ConversationBuffer:
        buf = self.buffers.get(channel_id)
        if not buf:
            buf = ConversationBuffer(limit_chars=self.limit_chars)
            self.buffers[channel_id] = buf
        return buf

    def add_user_message(self, channel_id: int, author_id: int, author_name: str, content: str):
        entry = Entry(
            role="user",
            author_id=author_id,
            author_name=author_name,
            content=content or "",
            created_at=datetime.utcnow(),
        )
        self._buf(channel_id).add(entry)

    def add_assistant_message(self, channel_id: int, content: str):
        entry = Entry(
            role="assistant",
            author_id=0,
            author_name="Lio",
            content=content or "",
            created_at=datetime.utcnow(),
        )
        self._buf(channel_id).add(entry)

    def export_chat_messages(self, channel_id: int, include_author_names: bool = True) -> List[ChatMessage]:
        return self._buf(channel_id).export(include_author_names=include_author_names)
