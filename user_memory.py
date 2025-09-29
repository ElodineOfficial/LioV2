# user_memory.py
from __future__ import annotations

import os
import re
import json
import logging
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

# Reuse your LLM provider + message type
from chatter import LLMProvider, ChatMessage


def _slugify(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", (name or "").strip())
    return s.strip("_") or "user"


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


@dataclass
class MemoryConfig:
    dir: str = "user_memory"
    max_files: int = 30
    max_facts: int = 10
    sweep_chars: int = 10_000
    index_path: str = "user_memory/index.json"


class UserMemoryStore:
    """
    User-ID based memory files with display-name fallback.

    - Stores at most `max_files` user .txt files in `dir/`
    - Each file contains up to `max_facts` lines (most-recent first)
    - Runs a memory sweep only when `seen_chars - swept_at >= sweep_chars`
    - Facts are extracted by the LLM (provider.complete_chat)
    """

    def __init__(self, provider: LLMProvider, config: MemoryConfig | None = None):
        self.provider = provider
        self.cfg = config or MemoryConfig()
        os.makedirs(self.cfg.dir, exist_ok=True)
        self.index: Dict[str, Dict] = self._load_index()

    # ---------- Public API ----------

    def note_user_text(self, user_id: Optional[int], display_name: str, text: str) -> None:
        """Increment per-user character counters. Call this for EVERY user message."""
        key = self._key(user_id, display_name)
        rec = self.index.setdefault(
            key,
            {"display_name": display_name, "user_id": user_id, "seen_chars": 0, "swept_at": 0},
        )
        rec["display_name"] = display_name or rec.get("display_name")  # refresh latest name
        rec["seen_chars"] += len(text or "")
        self._save_index()

    def maybe_sweep(
        self,
        history: List[ChatMessage],
        user_id: Optional[int],
        display_name: str,
    ) -> Optional[List[str]]:
        """
        If this user has accrued another `sweep_chars` since the last sweep, run an
        LLM pass to extract up to `max_facts` 'important' facts and persist them.
        Returns the merged facts if updated, else None.
        """
        key = self._key(user_id, display_name)
        rec = self.index.setdefault(
            key,
            {"display_name": display_name, "user_id": user_id, "seen_chars": 0, "swept_at": 0},
        )
        if (rec["seen_chars"] - rec.get("swept_at", 0)) < self.cfg.sweep_chars:
            return None

        # Build a context window (favor recency; keep size reasonable)
        context = self._build_context_for_user(history, target_display_name=display_name, max_chars=12_000)

        # Extract facts with the LLM
        facts = self._extract_facts(context=context, display_name=display_name, user_id=user_id, k=self.cfg.max_facts)

        # Persist / merge if we got anything back
        path = self._path_for(key)
        if facts:
            current = self._read_lines(path)
            merged = self._merge_facts(current, facts, self.cfg.max_facts)
            self._write_lines(path, merged)
            self._prune_file_count_if_needed(keep=path)
        else:
            merged = None

        # Mark sweep point
        rec["swept_at"] = rec["seen_chars"]
        self._save_index()
        return merged

    def get_memory_lines(self, user_id: Optional[int], display_name: str) -> List[str]:
        """Read up to `max_facts` lines for this user (empty list if none)."""
        return self._read_lines(self._path_for(self._key(user_id, display_name)))

    # ---------- Internals ----------

    def _key(self, user_id: Optional[int], display_name: str) -> str:
        return str(user_id) if user_id is not None else f"name:{_slugify(display_name)}"

    def _path_for(self, key: str) -> str:
        return os.path.join(self.cfg.dir, f"{key}.txt")

    def _load_index(self) -> Dict[str, Dict]:
        path = self.cfg.index_path
        try:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception:
            logging.exception("Failed reading memory index.")
        return {}

    def _save_index(self) -> None:
        path = self.cfg.index_path
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.index, f, ensure_ascii=False, indent=2)
            os.replace(tmp, path)
        except Exception:
            logging.exception("Failed writing memory index.")

    def _read_lines(self, path: str) -> List[str]:
        if not os.path.exists(path):
            return []
        try:
            with open(path, "r", encoding="utf-8") as f:
                # keep order as saved (most recent first)
                return [ln.rstrip("\n") for ln in f if ln.strip()]
        except Exception:
            logging.exception("Failed reading %s", path)
            return []

    def _write_lines(self, path: str, lines: List[str]) -> None:
        try:
            with open(path, "w", encoding="utf-8") as f:
                for ln in lines[: self.cfg.max_facts]:
                    f.write(ln.strip() + "\n")
        except Exception:
            logging.exception("Failed writing %s", path)

    def _merge_facts(self, current: List[str], new: List[str], max_facts: int) -> List[str]:
        # Most-recent-first, case/space-insensitive dedupe
        seen = set(_norm(x) for x in new)  # start by prioritizing NEW facts
        out = [x for x in new if _norm(x) in seen]  # preserve new order
        for x in current:
            nx = _norm(x)
            if nx not in seen:
                out.append(x)
                seen.add(nx)
            if len(out) >= max_facts:
                break
        return out[:max_facts]

    def _prune_file_count_if_needed(self, keep: str | None = None) -> None:
        try:
            files = [os.path.join(self.cfg.dir, f) for f in os.listdir(self.cfg.dir) if f.endswith(".txt")]
            if keep and keep in files:
                pass
            if len(files) <= self.cfg.max_files:
                return
            files_sorted = sorted(files, key=lambda p: os.path.getmtime(p))
            to_delete = []
            for p in files_sorted:
                if keep and p == keep:
                    continue
                to_delete.append(p)
                if len(files) - len(to_delete) <= self.cfg.max_files:
                    break
            for p in to_delete:
                try:
                    os.remove(p)
                except Exception:
                    logging.warning("Failed to prune old memory file: %s", p)
        except Exception:
            logging.exception("Failed memory file pruning.")

    def _build_context_for_user(
        self,
        history: List[ChatMessage],
        *,
        target_display_name: str,
        max_chars: int = 12_000,
    ) -> str:
        """
        Build a recent context chunk. We don't require author tagging in ChatMessage,
        so we pass the whole recent history and tell the LLM to extract only facts
        ABOUT the target user.
        """
        buf: List[str] = []
        running = 0
        # Use most recent messages first (history assumed chronological)
        for m in reversed(history):
            text = (m.content or "").strip()
            if not text:
                continue
            if running + len(text) + 1 > max_chars:
                break
            buf.append(text)
            running += len(text) + 1
        return "\n".join(reversed(buf))

    def _extract_facts(
        self,
        *,
        context: str,
        display_name: str,
        user_id: Optional[int],
        k: int,
    ) -> List[str]:
        if not context:
            return []

        system_prompt = (
            "You are a memory extractor for a Discord assistant. "
            "Given a chat transcript and a TARGET USER, write up to N durable, concise facts "
            "about that TARGET USER only. Prefer biographical details, stable preferences, "
            "and multi-message patterns. Ignore fleeting mood, one-off jokes, bot commands, "
            "and anything about other people. Avoid private/sensitive data (passwords, addresses). "
            "Output rules: plain text, one fact per line, no numbering/bullets, no extra text."
        )
        user_prompt = (
            f"N (max facts): {k}\n"
            f"TARGET USER: {display_name} (id={user_id if user_id is not None else 'unknown'})\n"
            "TRANSCRIPT:\n"
            f"{context}\n"
            "\nFacts:"
        )
        try:
            raw = self.provider.complete_chat(
                system_prompt=system_prompt,
                messages=[ChatMessage(role="user", content=user_prompt)],
                temperature=0.0,
                max_tokens=256,
            )
        except Exception:
            logging.exception("Memory extraction LLM call failed.")
            return []

        # Parse lines; sanitize; trim
        lines = [ln.strip(" -•\t") for ln in (raw or "").splitlines()]
        facts = [ln for ln in lines if ln and not ln.lower().startswith(("facts:", "fact:", "output:"))]
        # De-emoji / control chars (optional safety)
        facts = [re.sub(r"[\uFE0F\u200D]", "", f) for f in facts]
        # Keep only up to k lines
        return facts[:k]
