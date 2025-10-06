# bonk.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set, Optional

# We only rely on the public shape of ChatMessage (role, content).
# See chatter.ChatMessage in your repo. Roles are "user" | "assistant".
from chatter import ChatMessage  # type: ignore


@dataclass
class BonkRule:
    """Represents a single bonk event."""
    drop: int                      # how many assistant msgs to forget
    assistant_count_before: int    # number of assistant msgs that existed at bonk time


class BonkManager:
    """
    Keeps per-channel 'forget the last N assistant messages' rules.

    Design:
    - On `apply_bonk`, we snapshot how many assistant messages exist *at that moment*.
      We then forget the last `drop` among those *earlier* assistant messages.
      Any assistant messages sent after the bonk (like the ack) are unaffected.
    - `filter_history` takes the raw exported history and returns a pruned copy
      with the union of all bonk windows removed (assistant-only).
    """
    def __init__(self) -> None:
        self._rules: Dict[int, List[BonkRule]] = {}  # channel_id -> rules

    def apply_bonk(self, channel_id: int, history: List[ChatMessage], drop: int = 10) -> int:
        """Record a bonk for this channel. Returns how many assistant messages will be forgotten."""
        drop = int(drop) if drop else 10
        if drop < 1:
            drop = 1
        if drop > 50:
            drop = 50  # keep it sane

        asst_indices = [i for i, m in enumerate(history) if (m.role or "").lower() == "assistant"]
        assistant_count_before = len(asst_indices)
        effective = min(drop, assistant_count_before)

        # No assistants yet? Nothing to forget.
        if effective <= 0:
            # Still record an empty rule? Not necessary.
            return 0

        rules = self._rules.setdefault(channel_id, [])
        rules.append(BonkRule(drop=effective, assistant_count_before=assistant_count_before))
        return effective

    def filter_history(self, channel_id: int, history: List[ChatMessage]) -> List[ChatMessage]:
        """
        Return a new history with bonked assistant messages removed.
        Non-destructive: original history is not modified.
        """
        rules = self._rules.get(channel_id)
        if not rules:
            return history

        # Map assistant-order to absolute indices in the *current* history.
        asst_indices = [i for i, m in enumerate(history) if (m.role or "").lower() == "assistant"]
        total_asst = len(asst_indices)

        to_drop: Set[int] = set()

        for rule in rules:
            # Only consider assistant messages that existed *when the rule was created*.
            cap = min(rule.assistant_count_before, total_asst)
            if cap <= 0 or rule.drop <= 0:
                continue
            # Select the last `rule.drop` assistants among the first `cap` assistant messages.
            start = max(0, cap - rule.drop)
            victim_asst_indices = asst_indices[start:cap]
            to_drop.update(victim_asst_indices)

        if not to_drop:
            return history

        # Build filtered copy (preserving order).
        return [m for i, m in enumerate(history) if i not in to_drop]

    # Optional: if you later add a true deletion method on your store, you can call it here.
    def try_purge_store(self, store, channel_id: int, drop: int) -> bool:
        """
        If your ConversationStore ever exposes a deletion API (e.g., purge_last_assistant_messages),
        you can call it from harness via this helper. Returns True if a purge method was found and called.
        """
        method_name_candidates = [
            "purge_last_assistant_messages",    # proposed clear API
            "delete_last_assistant_messages",   # alternative naming
        ]
        for name in method_name_candidates:
            if hasattr(store, name):
                try:
                    getattr(store, name)(channel_id=channel_id, count=drop)
                    return True
                except Exception:
                    return False
        return False
