from __future__ import annotations

import os
import re
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Dict

from chatter import LLMProvider, ChatMessage

BRACKET_RE = re.compile(r"\[([^\[\]]+)\]")  # first [...] only


@dataclass
class OverheadDecision:
    route: str              # 'A' | 'B' | 'C' | 'GIF'
    emoji: Optional[str]    # unicode or '<:name:id>' or None
    gif_query: Optional[str] = None
    raw_command: str = ""


class Overhead:
    """
    Calls an LLM to return exactly one bracketed command:
      [A EMOJI=:sparkles:]
      [B EMOJI=none]
      [C EMOJI=<:lioLaugh:12345>]
      [GIF Q="cat vibing" EMOJI=none]
    """
    def __init__(
        self,
        provider: LLMProvider,
        emoji_path: str = "emoji.txt",
        log_path: str = "overhead_log.jsonl",
        max_log_lines: int = 2000,
        max_emoji_in_prompt: int = 100,
    ):
        self.provider = provider
        self.emoji_path = emoji_path
        self.log_path = log_path
        self.max_log_lines = max_log_lines
        self.max_emoji_in_prompt = max_emoji_in_prompt

        self._emoji_list: List[str] = self._load_emoji_file(emoji_path)
        self._route_history: Dict[int, List[str]] = {}
        self._emoji_history: Dict[int, List[str]] = {}

    def decide(
        self,
        *,
        channel_id: int,
        user_name: str,
        user_text: str,
        transcript_excerpt: str,
    ) -> OverheadDecision:
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(
            channel_id=channel_id,
            user_name=user_name,
            user_text=user_text,
            transcript_excerpt=transcript_excerpt,
        )
        raw = self.provider.complete_chat(
            system_prompt=system_prompt,
            messages=[ChatMessage(role="user", content=user_prompt)],
            temperature=0.6,
            max_tokens=64,
        )
        decision = self._parse_command(raw, channel_id=channel_id)
        self._log_decision(
            channel_id=channel_id,
            user_name=user_name,
            user_text=user_text,
            transcript_excerpt=transcript_excerpt,
            raw=raw,
            decision=decision,
        )
        self._push_history(channel_id, decision)
        return decision

    # ----- prompt -----

    def _build_system_prompt(self) -> str:
        sample_emojis = self._emoji_list[: self.max_emoji_in_prompt]
        emoji_hint = ", ".join(sample_emojis) if sample_emojis else "(none found)"
        return (
            "You are the routing brain for Lio. Return EXACTLY ONE command in square brackets, no commentary.\n"
            "Routes: A (friend/snark), B (DJ), C (helper), GIF (reaction gif).\n"
            "Grammar:\n"
            "  [A EMOJI=<emoji>|none]\n"
            "  [B EMOJI=<emoji>|none]\n"
            "  [C EMOJI=<emoji>|none]\n"
            '  [GIF Q="<search terms>" EMOJI=<emoji>|none]\n\n'
            "Constraints:\n"
            "- If you choose EMOJI, it MUST be one of the allowed emojis listed below (verbatim) or 'none'.\n"
            "- Only one bracketed command. No prose.\n"
            "- Prefer A for casual banter; B for music/DJ control; C for help/explanations; GIF when a reaction image fits best.\n"
            "- Keep variety; avoid repeating the same route/emoji excessively.\n\n"
            f"Allowed emojis: {emoji_hint}\n"
        )

    def _build_user_prompt(
        self,
        *,
        channel_id: int,
        user_name: str,
        user_text: str,
        transcript_excerpt: str,
    ) -> str:
        last_routes = " > ".join(self._route_history.get(channel_id, [])[-5:]) or "(none)"
        last_emojis = " > ".join(self._emoji_history.get(channel_id, [])[-5:]) or "(none)"
        return (
            f"User: {user_name}\n"
            f"Message: {user_text}\n"
            f"Recent context (trimmed): {transcript_excerpt}\n"
            f"Recent routes: {last_routes}\n"
            f"Recent emojis: {last_emojis}\n"
            "Choose one best route now and (optionally) an emoji."
        )

    # ----- parse -----

    def _parse_command(self, raw: str, channel_id: int) -> OverheadDecision:
        match = BRACKET_RE.search(raw or "")
        payload = match.group(1).strip() if match else ""
        tokens = self._tokenize(payload)

        route = "A"
        emoji: Optional[str] = None
        gif_query: Optional[str] = None

        if tokens:
            head = tokens.pop(0).upper()
            if head in {"A", "B", "C", "GIF"}:
                route = head

        params = self._parse_params(tokens)

        emo_raw = params.get("EMOJI")
        if emo_raw and emo_raw.lower() != "none":
            if emo_raw in self._emoji_list:
                emoji = emo_raw
            else:
                found = next((e for e in self._emoji_list if e.lower() == emo_raw.lower()), None)
                emoji = found

        if route == "GIF":
            gif_query = params.get("Q") or params.get("QUERY") or "reaction"

        recent = self._route_history.get(channel_id, [])[-5:]
        if route in {"A", "B", "C"} and all(r == route for r in recent) and route != "A":
            route = "A"

        return OverheadDecision(route=route, emoji=emoji, gif_query=gif_query, raw_command=payload)

    def _tokenize(self, payload: str) -> List[str]:
        if not payload:
            return []
        parts: List[str] = []
        buf = ""
        in_quotes = False
        for ch in payload:
            if ch == '"':
                in_quotes = not in_quotes
                buf += ch
            elif ch.isspace() and not in_quotes:
                if buf:
                    parts.append(buf)
                    buf = ""
            else:
                buf += ch
        if buf:
            parts.append(buf)
        return parts

    def _parse_params(self, tokens: List[str]) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for t in tokens:
            if "=" not in t:
                continue
            k, v = t.split("=", 1)
            k = k.strip().upper()
            v = v.strip()
            if len(v) >= 2 and v[0] == '"' and v[-1] == '"':
                v = v[1:-1]
            out[k] = v
        return out

    # ----- logging/variety -----

    def _log_decision(
        self,
        *,
        channel_id: int,
        user_name: str,
        user_text: str,
        transcript_excerpt: str,
        raw: str,
        decision: OverheadDecision,
    ):
        record = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "channel_id": channel_id,
            "user": user_name,
            "user_text": user_text,
            "context_excerpt": transcript_excerpt,
            "raw_overhead_output": raw,
            "parsed": {
                "route": decision.route,
                "emoji": decision.emoji,
                "gif_query": decision.gif_query,
                "raw_command": decision.raw_command,
            },
        }
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            logging.exception("Failed writing overhead log to %s", self.log_path)
        self._trim_log_if_needed()

    def _trim_log_if_needed(self):
        try:
            if not os.path.exists(self.log_path):
                return
            with open(self.log_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            if len(lines) <= self.max_log_lines:
                return
            tail = lines[-self.max_log_lines :]
            with open(self.log_path, "w", encoding="utf-8") as f:
                f.writelines(tail)
        except Exception:
            logging.exception("Failed trimming overhead log.")

    def _push_history(self, channel_id: int, decision: OverheadDecision):
        # Always ensure the lists exist for this channel
        routes = self._route_history.setdefault(channel_id, [])
        emojis = self._emoji_history.setdefault(channel_id, [])

        routes.append(decision.route)
        if decision.emoji:
            emojis.append(decision.emoji)

        # Trim to last 20 safely
        self._route_history[channel_id] = routes[-20:]
        self._emoji_history[channel_id] = emojis[-20:]

    # ----- emoji file -----

    def _load_emoji_file(self, path: str) -> List[str]:
        if not os.path.exists(path):
            return []
        out: List[str] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                out.append(s)
        # Dedup preserve order
        seen = set()
        uniq = []
        for e in out:
            if e not in seen:
                uniq.append(e)
                seen.add(e)
        return uniq

    # ----- NEW: emoji helpers for random reactions -----

    def pick_random_emoji(self) -> Optional[str]:
        """
        Return a random emoji from the configured collection (emoji.txt).
        Prefers Unicode emojis to maximize reaction success on any server,
        but will fall back to custom-emoji entries if no Unicode is present.
        """
        try:
            import random
            if not self._emoji_list:
                return None
            unicode_pool = [e for e in self._emoji_list if not e.startswith("<")]
            pool = unicode_pool if unicode_pool else self._emoji_list
            return random.choice(pool)
        except Exception:
            return None
