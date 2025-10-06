from __future__ import annotations

import os
import json
import random
import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional, Any, Dict, List

try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except Exception as e:
    raise SystemExit("Python 3.9+ with zoneinfo is required.") from e


@dataclass
class DailyPostConfig:
    posts_path: str = "daily_posts.json"
    state_path: str = "daily_post_state.json"
    enabled: bool = True
    min_delay_sec: int = 300   # 5 minutes
    max_delay_sec: int = 600   # 10 minutes
    cooldown_hours: int = 24   # extra guard: no two posts within 24h
    tz_env_var: str = "TZ"     # read local TZ from here (falls back to America/New_York)


class DailyPostManager:
    """
    Once per *calendar day*, and also not more than once per 24h, post a prewritten line from a local JSON.

    JSON format supported:
      - Simple list of strings: ["one", "two", ...]
      - List of objects: [{"id": "2025-001", "text": "Hi"}, ...]
      - Object with 'posts': { "posts": [ ... as above ... ] }

    Persistence:
      - state JSON tracks:
            {
              "last_post_utc": "YYYY-MM-DDTHH:MM:SSZ",
              "last_post_local_date": "YYYY-MM-DD",
              "last_post_id": "optional",
              "next_index": 0,
              "last_channel_id": 1234567890
            }

    Usage:
      manager = DailyPostManager(...)
      await manager.schedule_once(channel=my_discord_channel, store=ConversationStore, sanitize=strip_fn)
    """

    def __init__(self, config: DailyPostConfig):
        self.cfg = config
        self._task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
        self._sanitize: Optional[Callable[[str], str]] = None

    def set_sanitizer(self, fn: Optional[Callable[[str], str]]) -> None:
        """Optional hook to scrub or format text before posting (e.g., remove inline emojis)."""
        self._sanitize = fn

    # ---------- Public API ----------

    async def schedule_once(self, channel: Any, store: Optional[Any] = None, sanitize: Optional[Callable[[str], str]] = None) -> None:
        """
        Schedule a single post attempt after a random delay within [min_delay_sec, max_delay_sec].
        No-op if disabled or an existing schedule is active.
        """
        if not self.cfg.enabled:
            logging.info("DailyPost: disabled by config.")
            return
        if sanitize is not None:
            self._sanitize = sanitize
        if self._task and not self._task.done():
            logging.info("DailyPost: schedule already active; skipping duplicate.")
            return
        try:
            delay = random.randint(max(0, self.cfg.min_delay_sec), max(0, self.cfg.max_delay_sec))
            if delay < 0:
                delay = 0
        except Exception:
            delay = 300
        logging.info("DailyPost: scheduled in %s seconds.", delay)
        self._task = asyncio.create_task(self._run_after_delay(delay, channel, store))

    async def maybe_post_now(self, channel: Any, store: Optional[Any] = None) -> bool:
        """
        Attempt the post immediately (observes calendar-day + 24h guards).
        Returns True if a post was made; False otherwise.
        """
        async with self._lock:
            if not self.cfg.enabled:
                logging.info("DailyPost: disabled by config.")
                return False

            posts = self._load_posts()
            if not posts:
                logging.warning("DailyPost: no posts found at %s", self.cfg.posts_path)
                return False

            state = self._load_state()
            if not self._allowed_to_post(state):
                logging.info("DailyPost: guard prevented posting (calendar/24h).")
                return False

            idx = self._resolve_next_index(state, len(posts))
            post = self._pick_post(posts, idx)
            if not post:
                logging.warning("DailyPost: could not resolve a valid post.")
                return False

            text = post.get("text", "").strip()
            if not text:
                logging.warning("DailyPost: selected post had empty text; skipping.")
                return False

            # Optional sanitize (e.g., remove inline emojis)
            if self._sanitize:
                try:
                    text = self._sanitize(text) or text
                except Exception:
                    pass

            # Send to Discord channel
            try:
                await channel.send(text)
            except Exception:
                logging.exception("DailyPost: failed to send message to channel.")
                return False

            # Persist to conversation store if provided
            try:
                if store is not None:
                    # store.add_assistant_message(channel_id=channel.id, content=text)
                    # ConversationStore signature in your code uses channel_id; guard for attribute presence.
                    ch_id = getattr(channel, "id", None)
                    if ch_id is not None and hasattr(store, "add_assistant_message"):
                        store.add_assistant_message(channel_id=ch_id, content=text)
            except Exception:
                logging.exception("DailyPost: failed to persist assistant message to store.")

            # Update state after successful send
            now_utc = datetime.now(timezone.utc)
            local_date = self._now_local().date().isoformat()
            state["last_post_utc"] = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
            state["last_post_local_date"] = local_date
            state["last_post_id"] = post.get("id")
            state["last_channel_id"] = getattr(channel, "id", None)
            state["next_index"] = (idx + 1) % len(posts)
            self._save_state(state)

            logging.info("DailyPost: posted id=%s; next_index=%s", post.get("id"), state["next_index"])
            return True

    # ---------- Internal helpers ----------

    async def _run_after_delay(self, delay: int, channel: Any, store: Optional[Any]) -> None:
        try:
            await asyncio.sleep(delay)
            await self.maybe_post_now(channel, store)
        except Exception:
            logging.exception("DailyPost: error in delayed run.")

    def _now_local(self) -> datetime:
        tz_name = os.getenv(self.cfg.tz_env_var, "America/New_York")
        try:
            tz = ZoneInfo(tz_name)
        except Exception:
            logging.warning("DailyPost: invalid TZ '%s'; using America/New_York", tz_name)
            tz = ZoneInfo("America/New_York")
        return datetime.now(tz)

    def _load_posts(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.cfg.posts_path):
            return []
        try:
            with open(self.cfg.posts_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except Exception:
            logging.exception("DailyPost: failed to load posts JSON.")
            return []

        # Normalize into list[dict{id?, text, enabled?}]
        posts_in: List[Any]
        if isinstance(raw, dict) and "posts" in raw:
            posts_in = raw.get("posts") or []
        elif isinstance(raw, list):
            posts_in = raw
        else:
            return []

        posts_out: List[Dict[str, Any]] = []
        for i, item in enumerate(posts_in):
            if isinstance(item, str):
                posts_out.append({"id": f"auto-{i}", "text": item, "enabled": True})
                continue
            if isinstance(item, dict):
                text = str(item.get("text", "")).strip()
                if not text:
                    continue
                enabled = item.get("enabled", True)
                pid = item.get("id", f"auto-{i}")
                posts_out.append({"id": pid, "text": text, "enabled": enabled})
        # Keep only enabled
        posts_out = [p for p in posts_out if p.get("enabled", True)]
        return posts_out

    def _load_state(self) -> Dict[str, Any]:
        if not os.path.exists(self.cfg.state_path):
            return {}
        try:
            with open(self.cfg.state_path, "r", encoding="utf-8") as f:
                return json.load(f) or {}
        except Exception:
            logging.exception("DailyPost: failed to read state file.")
            return {}

    def _save_state(self, state: Dict[str, Any]) -> None:
        try:
            with open(self.cfg.state_path, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
        except Exception:
            logging.exception("DailyPost: failed to write state file.")

    def _resolve_next_index(self, state: Dict[str, Any], n: int) -> int:
        idx = state.get("next_index", 0)
        try:
            idx = int(idx)
        except Exception:
            idx = 0
        if n <= 0:
            return 0
        return idx % n

    def _pick_post(self, posts: List[Dict[str, Any]], idx: int) -> Optional[Dict[str, Any]]:
        if not posts:
            return None
        if idx < 0 or idx >= len(posts):
            idx = 0
        return posts[idx]

    def _allowed_to_post(self, state: Dict[str, Any]) -> bool:
        """
        True iff (a) we have not posted yet today in local calendar AND
                 (b) at least cooldown_hours have elapsed since last UTC post.
        If there's no state, allow posting.
        """
        # Calendar-day check
        last_local_date = state.get("last_post_local_date")
        today = self._now_local().date().isoformat()
        if last_local_date == today:
            return False

        # 24-hour check
        last_utc = state.get("last_post_utc")
        if last_utc:
            try:
                if last_utc.endswith("Z"):
                    last_dt = datetime.strptime(last_utc, "%Y-%m-%dT%H:%M:%SZ")
                    last_dt = last_dt.replace(tzinfo=timezone.utc)
                else:
                    last_dt = datetime.fromisoformat(last_utc)
                    if last_dt.tzinfo is None:
                        last_dt = last_dt.replace(tzinfo=timezone.utc)
                elapsed = datetime.now(timezone.utc) - last_dt
                if elapsed < timedelta(hours=max(0, self.cfg.cooldown_hours)):
                    return False
            except Exception:
                # If parse fails, fall through to allow post (better to post than to fail silently forever)
                pass
        return True
