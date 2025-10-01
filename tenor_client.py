# tenor_client.py
from __future__ import annotations

import os
import logging
import random
import asyncio
from typing import Optional, Dict, List
from collections import deque

try:
    import aiohttp  # type: ignore
except Exception:
    aiohttp = None


class TenorClient:
    """
    Minimal Tenor v2 client with de-dup + variety.

    Behavior:
      - Keeps a small rolling history of recently served GIF URLs *per query*.
      - Randomly selects from a batch of results while excluding the recent ones.
      - Always prefixes the search query with 'anime ' before calling Tenor, so
        'dance' becomes 'anime dance'. If the query already starts with 'anime '
        (case-insensitive), it is left as-is.

    Env toggles (all optional):
      TENOR_API_KEY           : your key (same as before)
      GIF_SEARCH_LIMIT        : how many results to fetch per search (default: 20)
      GIF_RECENT_BLOCK        : how many recently-used GIFs (per query) to avoid (default: 5)
      TENOR_CONTENT_FILTER    : Tenor content filter (default: "high")
      TENOR_RANDOM            : "true"/"false" to enable Tenor's randomization (default: "true")
    """

    def __init__(self, api_key: Optional[str] = None,
                 recent_block: Optional[int] = None,
                 search_limit: Optional[int] = None):
        self.api_key = api_key or os.getenv("TENOR_API_KEY")
        # Configurable knobs with sensible defaults
        self.search_limit = int(os.getenv("GIF_SEARCH_LIMIT", str(search_limit or 20)))
        self.recent_block = int(os.getenv("GIF_RECENT_BLOCK", str(recent_block or 5)))
        self.content_filter = os.getenv("TENOR_CONTENT_FILTER", "high")
        self.randomize = (os.getenv("TENOR_RANDOM", "true").lower() == "true")

        # Per-query rolling history of served URLs
        self._recent_by_query: Dict[str, deque[str]] = {}
        self._lock = asyncio.Lock()

    def available(self) -> bool:
        return bool(self.api_key and aiohttp is not None)

    def _ensure_anime_prefix(self, query: str) -> str:
        """
        Ensure the query begins with 'anime '.
        If the input is empty/whitespace, return 'anime reaction'.
        """
        q = (query or "").strip()
        if not q:
            return "anime reaction"
        if q.lower().startswith("anime "):
            return q
        return f"anime {q}"

    async def search_first_gif_url(self, query: str) -> Optional[str]:
        """
        Previous behavior: "first result wins".
        New behavior: fetch a batch, filter out last N used for this query, pick randomly.
        Keeps method name/signature for drop-in compatibility.
        Additionally, ALWAYS prefixes 'anime ' to the provided query.
        """
        if not self.available():
            return None

        prefixed_query = self._ensure_anime_prefix(query)

        base = "https://tenor.googleapis.com/v2/search"
        params = {
            "q": prefixed_query,
            "key": self.api_key,
            "limit": self.search_limit,
            "media_filter": "gif,tinygif,mp4",
            "random": "true" if self.randomize else "false",
            "contentfilter": self.content_filter,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(base, params=params, timeout=10) as resp:
                    if resp.status != 200:
                        logging.warning("Tenor response %s for query '%s'", resp.status, prefixed_query)
                        return None
                    data = await resp.json()
        except Exception:
            logging.exception("Tenor request failed for query '%s'", prefixed_query)
            return None

        results = data.get("results") or []
        if not results:
            return None

        # Build a candidate list of URLs; prefer one media per result to increase variety across items
        candidates: List[str] = []
        for item in results:
            media = item.get("media_formats") or {}
            # Prefer gif > tinygif > mp4
            for key in ("gif", "tinygif", "mp4"):
                url = (media.get(key) or {}).get("url")
                if url and url not in candidates:
                    candidates.append(url)
                    break

        if not candidates:
            return None

        # De-dup against the last `recent_block` picks for THIS (prefixed) query
        norm_q = (prefixed_query or "").strip().lower() or "<blank>"
        async with self._lock:
            dq = self._recent_by_query.get(norm_q)
            if dq is None or dq.maxlen != self.recent_block:
                dq = deque(maxlen=max(0, self.recent_block))
                self._recent_by_query[norm_q] = dq

            pool = [u for u in candidates if u not in dq] if dq.maxlen > 0 else list(candidates)
            if not pool:
                # Nothing new? fall back to full candidate list to avoid hard failures
                pool = candidates

            choice = random.choice(pool)
            if dq.maxlen > 0:
                dq.append(choice)

        return choice
