# harness.py
import os
import re
import logging
import asyncio
import mimetypes
from datetime import datetime
from typing import List, Tuple, Optional

import discord
from dotenv import load_dotenv

from chat_store import ConversationStore
from chatter import Chatter, build_provider
from overhead import Overhead, OverheadDecision
from tenor_client import TenorClient
from user_memory import UserMemoryStore, MemoryConfig

# NEW: daily post manager
from daily_post import DailyPostManager, DailyPostConfig

# <<< BONK
from bonk import BonkManager

# NEW: Gemini vision summarizer
from vision import GeminiVision

try:
    from zoneinfo import ZoneInfo
except ImportError:
    raise SystemExit("Python 3.9+ with zoneinfo is required; use Python 3.10+ for best results.")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ----------------- Time -----------------

def now_eastern() -> datetime:
    tz_name = os.getenv("TZ", "America/New_York")
    try:
        tz = ZoneInfo(tz_name)
    except Exception:
        logging.warning("Invalid TZ '%s'; falling back to America/New_York", tz_name)
        tz = ZoneInfo("America/New_York")
    return datetime.now(tz=tz)

# ----------------- Discord Bot -----------------

class LioHarness(discord.Client):
    def __init__(self, *, channel_id: int | None, channel_name: str,
                 chatter: Chatter, store: ConversationStore,
                 overhead: Overhead, tenor: TenorClient,
                 user_memory: UserMemoryStore,
                 daily_post: DailyPostManager | None = None,
                 cv: GeminiVision | None = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.channel_id = channel_id
        self.channel_name = channel_name
        self.chatter = chatter
        self.store = store
        self.overhead = overhead
        self.tenor = tenor
        self.user_memory = user_memory
        self.daily_post = daily_post
        self._announced = False

        # <<< BONK
        self.bonk = BonkManager()

        # Keep GIF behavior; no CV.
        self.gif_random_prob = float(os.getenv("GIF_RANDOM_PROB", "0.05"))

        # UPDATED: default to a 20-30% chance to react to the user's message after replies.
        self.react_to_user_prob = self._parse_prob(
            os.getenv("REACT_TO_USER_PROB", "0.3"), default=0.3
        )

        # Probability of posting the emoji as a *separate* message so it renders large.
        # Accepts "1:3", "1/3", or a decimal like "0.3333". Default is "1:9".
        self.emoji_second_post_prob = self._parse_prob(
            os.getenv("EMOJI_SECOND_POST_PROB", "1:9"), default=1.0 / 9.0
        )

        # Emoji scrubbing for replies
        self._custom_emoji_re = re.compile(r"<a?:[A-Za-z0-9_]+:\d+>")
        self._colon_code_re = re.compile(r":[A-Za-z0-9_+\-]{1,64}:")
        self._vs_zwj_re = re.compile(r"[\uFE0F\u200D]")
        self._emoji_unicode_re = re.compile(
            "("
            r"[\U0001F1E6-\U0001F1FF]"
            r"|[\U0001F300-\U0001F5FF]"
            r"|[\U0001F600-\U0001F64F]"
            r"|[\U0001F680-\U0001F6FF]"
            r"|[\U0001F700-\U0001F77F]"
            r"|[\U0001F780-\U0001F7FF]"
            r"|[\U0001F800-\U0001F8FF]"
            r"|[\U0001F900-\U0001F9FF]"
            r"|[\U0001FA00-\U0001FA6F]"
            r"|[\U0001FA70-\U0001FAFF]"
            r"|[\u2600-\u26FF]"
            r"|[\u2700-\u27BF]"
            ")"
        )

        # NEW: Global single-flight gate for mention-directed processing.
        # Ensures we never process more than one @-message at the same time.
        self._mention_gate: asyncio.Lock = asyncio.Lock()

        # --- CV controls ---
        self.cv = cv
        self.cv_enabled = (os.getenv("CV_ENABLED", "1").strip() != "0")
        self.cv_mentions_only = (os.getenv("CV_MENTIONS_ONLY", "1").strip() != "0")

    async def on_ready(self):
        if self._announced:
            return
        self._announced = True
        logging.info("Logged in as %s (%s)", self.user, self.user.id)
        channel = await self._resolve_channel()
        if not channel:
            logging.error("Could not resolve target channel '%s'.", self.channel_name)
            return
        ts = now_eastern().strftime("%Y-%m-%d %H:%M:%S %Z")
        msg = f"Reporting for duty! Connection established {ts}"
        try:
            await channel.send(msg)
            logging.info("Sent '%s' to #%s (%s)", msg, channel.name, channel.id)
        except Exception:
            logging.exception("Failed to send connection message.")

        # NEW: Schedule the once-per-day startup post
        try:
            if self.daily_post:
                # Let daily-post reuse this bot's emoji scrubber for consistency.
                self.daily_post.set_sanitizer(self._strip_emojis)
                await self.daily_post.schedule_once(channel=channel, store=self.store)
        except Exception:
            logging.exception("Failed to schedule daily post.")

    async def on_message(self, message: discord.Message):
        # Ignore our own messages
        if message.author == self.user:
            return
        # Only respond in the configured channel (or its threads)
        if not await self._is_target_channel(message.channel):
            return

        content = message.content or ""
        author_name = getattr(message.author, "display_name", message.author.name)
        stripped = (content or "").strip()

        # <<< BONK: handle command early (no mention required)
        if stripped.lower().startswith("!bonk"):
            # optional count: "!bonk 7"
            parts = stripped.split()
            count = 10
            if len(parts) >= 2:
                try:
                    count = max(1, min(50, int(parts[1])))
                except Exception:
                    count = 10

            # Grab current history snapshot to anchor the bonk window.
            try:
                raw_hist = self.store.export_chat_messages(
                    channel_id=message.channel.id, include_author_names=True
                )
            except Exception:
                raw_hist = []

            # If your store ever grows a deletion API, enable the hard purge here.
            hard_purged = False
            try:
                hard_purged = self.bonk.try_purge_store(self.store, message.channel.id, count)
            except Exception:
                hard_purged = False

            if hard_purged:
                ack = f"Bonk applied, try not to jostle my circuits. {count} messages purged."
            else:
                effective = self.bonk.apply_bonk(message.channel.id, raw_hist, count)
                if effective == 0:
                    ack = "Bonk applied, but I have no assistant messages to forget in this channel, why'd you even do that?"
                else:
                    ack = f"Bonk applied, try not to jostle my circuits. {count} messages purged."

            try:
                await message.channel.send(ack)
                # Record a minimal breadcrumb in the store; short and non-harmful for context.
                self.store.add_assistant_message(
                    channel_id=message.channel.id, content="[BONK] " + ack
                )
            except Exception:
                logging.exception("Failed to send bonk ack.")
            return
        # <<< BONK end

        # Determine whether this message targets the bot
        is_mention = (self.user in getattr(message, "mentions", []))
        is_reply_to_bot = False
        if message.reference and message.reference.resolved:
            try:
                is_reply_to_bot = getattr(message.reference.resolved, "author", None).id == self.user.id
            except Exception:
                is_reply_to_bot = False

        # For non-mention chatter, keep memory + transcript as before, but do not reply.
        if not (is_mention or is_reply_to_bot):
            try:
                self.user_memory.note_user_text(message.author.id, author_name, content)
            except Exception:
                logging.exception("note_user_text failed (non-mention)")
            try:
                self.store.add_user_message(
                    channel_id=message.channel.id,
                    author_id=message.author.id,
                    author_name=author_name,
                    content=content
                )
            except Exception:
                logging.exception("Failed to persist non-mention user message.")

            # Optional CV on passive messages (disabled by default)
            if self.cv and self.cv_enabled and not self.cv_mentions_only:
                try:
                    cv_text = await self._summarize_attachments_if_any(message, user_text=content)
                    if cv_text:
                        self.store.add_user_message(
                            channel_id=message.channel.id,
                            author_id=message.author.id,
                            author_name=author_name,
                            content=cv_text,
                        )
                except Exception:
                    logging.exception("CV summarize (non-mention) failed.")
            return

        # ------------------ SINGLE-FLIGHT SECTION ------------------
        # Only one @-directed message is processed at a time.
        async with self._mention_gate:
            # Memory tap for mention-directed messages
            try:
                self.user_memory.note_user_text(message.author.id, author_name, content)
            except Exception:
                logging.exception("note_user_text failed (mention)")

            # Persist the *mention* user message under the gate so history for this turn
            # doesn't include later mentions that arrived while we were busy.
            try:
                self.store.add_user_message(
                    channel_id=message.channel.id,
                    author_id=message.author.id,
                    author_name=author_name,
                    content=content
                )
            except Exception:
                logging.exception("Failed to persist mention user message.")

            # ---------------------------------------------------------------
            # Show native typing indicator for ALL processing on this turn.
            # ---------------------------------------------------------------
            decision: OverheadDecision | None = None
            reply_text: str | None = None

            try:
                async with message.channel.typing():
                    # === NEW: CV step ===
                    if self.cv and self.cv_enabled:
                        try:
                            cv_text = await self._summarize_attachments_if_any(message, user_text=content)
                            if cv_text:
                                # Persist the CV summary so it's visible in the exported history
                                self.store.add_user_message(
                                    channel_id=message.channel.id,
                                    author_id=message.author.id,
                                    author_name=author_name,
                                    content=cv_text,
                                )
                        except Exception:
                            logging.exception("CV summarize (mention) failed.")

                    # Build recent context AFTER the (optional) CV insert
                    history_raw = self.store.export_chat_messages(
                        channel_id=message.channel.id, include_author_names=True
                    )
                    # <<< BONK: filter assistant msgs per active bonks
                    history = self.bonk.filter_history(message.channel.id, history_raw)
                    excerpt = "\n".join(m.content for m in history)[-1000:]

                    # Route selection
                    decision = self.overhead.decide(
                        channel_id=message.channel.id,
                        user_name=author_name,
                        user_text=content,
                        transcript_excerpt=excerpt,
                    )

                    # Occasional reaction GIF (kept)
                    try:
                        import random
                        if decision.route != "GIF":
                            if random.random() < self.gif_random_prob:
                                decision = OverheadDecision(
                                    route="GIF",
                                    emoji=decision.emoji,
                                    gif_query=self._derive_gif_query(content),
                                    raw_command="RANDOM_GIF",
                                )
                    except Exception:
                        logging.exception("Random GIF overlay failed.")

                    # If GIF route, do the GIF work (including Tenor search) while typing
                    if decision.route == "GIF":
                        # Send GIF reply and maybe react to the user's message (probability gate)
                        await self._send_gif_reply(message, decision)
                        return  # gate released automatically on 'async with' exit

                    # Otherwise, generate text reply while typing
                    persona_map = {
                        "A": os.getenv("PERSONALITY_A_PATH", "personalityA.txt"),
                        "B": os.getenv("PERSONALITY_B_PATH", "personalityB.txt"),
                        "C": os.getenv("PERSONALITY_C_PATH", "personalityC.txt"),
                    }
                    system_prompt = self._load_text_file_safe(persona_map.get(decision.route, "personalityA.txt"))

                    try:
                        reply_text = await self.chatter.generate_reply_with_system_async(history, system_prompt)
                    except Exception:
                        logging.exception("Chatter generation failed.")
                        reply_text = "My brain farted. Try again?"
            except Exception:
                logging.exception("Unexpected failure in on_message processing (mention).")
                reply_text = "My brain farted. Try again?"

            # Send final output (typing indicator stops automatically here)
            # 1) Scrub any emojis from the LLM text to ensure emojis are never inline.
            text_only = self._strip_emojis(reply_text or "") or "…"

            try:
                await message.channel.send(text_only)
                self.store.add_assistant_message(channel_id=message.channel.id, content=text_only)
            except Exception:
                logging.exception("Failed to send persona reply.")

            # Maybe react to the user's message with a random emoji (probability gate).
            await self._react_randomly_to_user_message(message)

            # 2) Optionally post the chosen emoji as a separate follow-up (unchanged behavior).
            try:
                if decision and decision.emoji:
                    import random
                    if random.random() < self.emoji_second_post_prob:
                        await message.channel.send(decision.emoji)
                        self.store.add_assistant_message(
                            channel_id=message.channel.id, content=decision.emoji
                        )
            except Exception:
                logging.exception("Failed to send emoji as separate post.")

    # ---------- helpers ----------

    async def _summarize_attachments_if_any(self, message: discord.Message, *, user_text: str) -> Optional[str]:
        """
        If the message has image/* attachments, fetch bytes and ask Gemini for a tiny JSON summary.
        Returns a '[CV] {...}' string or None. Runs image fetch async; Gemini call in thread.
        """
        if not getattr(message, "attachments", None):
            return None

        # Collect (bytes, mime) for image attachments, up to CV_MAX_IMAGES
        images: List[Tuple[bytes, str]] = []
        for a in message.attachments:
            ct = (getattr(a, "content_type", None) or "").lower().strip()
            if not ct and getattr(a, "filename", None):
                guessed, _ = mimetypes.guess_type(a.filename)
                ct = (guessed or "").lower().strip()
            if not ct.startswith("image/"):
                continue
            try:
                # discord.py provides an async .read()
                data = await a.read()
                if data:
                    images.append((data, ct or "image/jpeg"))
            except Exception:
                logging.exception("Failed reading attachment bytes.")
                continue

        if not images:
            return None

        # Run the synchronous Gemini call off the event loop.
        try:
            return await asyncio.to_thread(self.cv.summarize, images, user_hint=user_text)
        except Exception:
            logging.exception("Gemini summarize in thread failed.")
            return None

    def _strip_emojis(self, text: str) -> str:
        if not text: return text
        s = text
        s = re.sub(r"<a?:[A-Za-z0-9_]+:\d+>", "", s)
        s = re.sub(r":[A-Za-z0-9_+\-]{1,64}:", "", s)
        s = re.sub(r"[\uFE0F\u200D]", "", s)
        s = re.sub("("
                   r"[\U0001F1E6-\U0001F1FF]"
                   r"|[\U0001F300-\U0001F5FF]"
                   r"|[\U0001F600-\U0001F64F]"
                   r"|[\U0001F680-\U0001F6FF]"
                   r"|[\U0001F700-\U0001F77F]"
                   r"|[\U0001F780-\U0001F7FF]"
                   r"|[\U0001F800-\U0001F8FF]"
                   r"|[\U0001F900-\U0001F9FF]"
                   r"|[\U0001FA00-\U0001FA6F]"
                   r"|[\U0001FA70-\U0001FAFF]"
                   r"|[\u2600-\u26FF]"
                   r"|[\u2700-\u27BF]"
                   ")", "", s)
        return re.sub(r"\s{2,}", " ", s).strip()

    def _load_text_file_safe(self, path: str) -> str:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read().strip()
        except FileNotFoundError:
            logging.warning("Personality file not found: %s", path)
            return ""

    def _derive_gif_query(self, text: str) -> str:
        text = (text or "").strip().lower()
        if not text:
            return "reaction"
        # Very small heuristic to keep GIFs relevant
        if "thanks" in text or "thank you" in text:
            return "thank you reaction"
        if "lol" in text or "funny" in text or "lmao" in text:
            return "lol reaction"
        if "brb" in text or "back later" in text:
            return "brb reaction"
        return "reaction"

    async def _send_gif_reply(self, source_message: discord.Message, decision: OverheadDecision):
        """
        Send the GIF reply to the channel, then maybe react to the *user's* original message
        with a random emoji (probability gate).
        """
        channel: discord.TextChannel = source_message.channel  # type: ignore[assignment]
        url = await self.tenor.search_first_gif_url(decision.gif_query or "reaction")
        if url:
            try:
                await channel.send(url)
                # Maybe react to the user's message with a random emoji (probability gate).
                await self._react_randomly_to_user_message(source_message)
                self.store.add_assistant_message(channel_id=channel.id, content=f"[GIF] {url}")
                return
            except Exception:
                logging.exception("Failed sending GIF.")
        # Fallback: textual hint if no GIF URL
        fallback = f"*{decision.gif_query or 'reaction'}*"
        try:
            await channel.send(fallback)
            await self._react_randomly_to_user_message(source_message)
            self.store.add_assistant_message(channel_id=channel.id, content=fallback)
        except Exception:
            logging.exception("Failed sending GIF fallback.")

    async def _maybe_react_to_user_message(self, user_message: discord.Message, emoji: str | None) -> None:
        """
        Legacy helper: used to react to the *user's* message with a chosen emoji and a probability gate.
        Kept for compatibility, but new code prefers _react_randomly_to_user_message.
        """
        try:
            import random
            if not emoji:
                return
            if random.random() >= self.react_to_user_prob:
                return
            from discord import PartialEmoji
            try:
                # Support both '<:name:id>' and '<a:name:id>' custom emoji strings via PartialEmoji
                await user_message.add_reaction(
                    PartialEmoji.from_str(emoji) if emoji.startswith("<") else emoji
                )
            except Exception:
                # Swallow reaction errors silently (missing perms, emoji not in guild, etc.)
                pass
        except Exception:
            logging.exception("Failed during maybe-react-to-user logic.")

    async def _react_randomly_to_user_message(self, user_message: discord.Message) -> None:
        """
        Attempt to react to the user's message with a random emoji from Overhead's collection,
        based on self.react_to_user_prob (default 20-30%). Prefers Unicode emojis for higher success;
        retries a few times if a chosen emoji fails.
        """
        try:
            import random
            if random.random() >= self.react_to_user_prob:
                return
            from discord import PartialEmoji
            attempts = 4
            for _ in range(attempts):
                emoji = self.overhead.pick_random_emoji()
                if not emoji:
                    return
                try:
                    await user_message.add_reaction(
                        PartialEmoji.from_str(emoji) if emoji.startswith("<") else emoji
                    )
                    return
                except Exception:
                    # Try another random emoji if this one isn't usable (e.g., custom emoji not in guild)
                    continue
        except Exception:
            logging.exception("Failed during random react-to-user logic.")

    def _parse_prob(self, s: str, default: float) -> float:
        """
        Parse a probability from strings like '1:3', '1/3', or '0.3333'.
        Returns 'default' on any error.
        """
        try:
            s = (s or "").strip()
            if ":" in s:
                a, b = s.split(":", 1)
                return float(a) / float(b)
            if "/" in s:
                a, b = s.split("/", 1)
                return float(a) / float(b)
            return float(s)
        except Exception:
            return default

    async def _resolve_channel(self) -> discord.TextChannel | None:
        if self.channel_id:
            ch = self.get_channel(self.channel_id)
            if isinstance(ch, discord.TextChannel): return ch
            try:
                fetched = await self.fetch_channel(self.channel_id)
                if isinstance(fetched, discord.TextChannel): return fetched
            except Exception:
                logging.exception("Failed to fetch channel by id: %s", self.channel_id)
        for guild in self.guilds:
            for ch in getattr(guild, "text_channels", []):
                if getattr(ch, "name", None) == self.channel_name:
                    return ch
        return None

    async def _is_target_channel(self, channel: discord.abc.Messageable) -> bool:
        try:
            if self.channel_id and getattr(channel, "id", None) == self.channel_id: return True
            if getattr(channel, "name", None) == self.channel_name: return True
            parent = getattr(channel, "parent", None)
            if parent and getattr(parent, "name", None) == self.channel_name: return True
        except Exception:
            pass
        return False

# ----------------- Entrypoint -----------------

def main():
    load_dotenv()

    token = os.getenv("DISCORD_TOKEN")
    if not token:
        raise SystemExit("Missing DISCORD_TOKEN in environment or .env")

    channel_id_env = os.getenv("LIOS_TEST_CHANNEL_ID")
    channel_id = int(channel_id_env) if channel_id_env else None
    channel_name = os.getenv("LIOS_TEST_CHANNEL_NAME", "lios-test-room")

    limit_chars = int(os.getenv("THREAD_CHAR_LIMIT", "10000"))
    store = ConversationStore(limit_chars=limit_chars)

    provider = build_provider()
    personality_path = os.getenv("PERSONALITY_A_PATH", "personalityA.txt")
    chatter = Chatter(provider=provider, personality_path=personality_path)

    mem_cfg = MemoryConfig(
        dir=os.getenv("MEMORY_DIR", "user_memory"),
        max_files=int(os.getenv("MEMORY_MAX_FILES", "30")),
        max_facts=int(os.getenv("MEMORY_MAX_FACTS", "10")),
        sweep_chars=int(os.getenv("MEMORY_SWEEP_CHARS", "10000")),
        index_path=os.getenv("MEMORY_INDEX_PATH", "user_memory/index.json"),
    )
    user_memory = UserMemoryStore(provider=provider, config=mem_cfg)

    overhead = Overhead(
        provider=provider,
        emoji_path=os.getenv("EMOJI_PATH", "emoji.txt"),
        log_path=os.getenv("OVERHEAD_LOG_PATH", "overhead_log.jsonl"),
        max_log_lines=int(os.getenv("OVERHEAD_LOG_MAX", "2000")),
    )
    tenor = TenorClient(api_key=os.getenv("TENOR_API_KEY"))

    # NEW: Daily post manager config from env
    daily_cfg = DailyPostConfig(
        posts_path=os.getenv("DAILY_POSTS_PATH", "daily_posts.json"),
        state_path=os.getenv("DAILY_POST_STATE_PATH", "daily_post_state.json"),
        enabled=(os.getenv("DAILY_POST_ENABLED", "1").strip() != "0"),
        min_delay_sec=int(os.getenv("DAILY_POST_MIN_DELAY_SEC", "300")),
        max_delay_sec=int(os.getenv("DAILY_POST_MAX_DELAY_SEC", "600")),
        cooldown_hours=int(os.getenv("DAILY_POST_COOLDOWN_HOURS", "24")),
        tz_env_var=os.getenv("DAILY_POST_TZ_ENV", "TZ")
    )
    daily_post = DailyPostManager(config=daily_cfg)

    # NEW: Vision (Gemini) – configurable on/off by env
    cv = None
    try:
        if os.getenv("CV_ENABLED", "1").strip() != "0":
            cv = GeminiVision(model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite"))
            logging.info("Gemini vision enabled (model=%s)", cv.model)
        else:
            logging.info("Gemini vision disabled via CV_ENABLED=0")
    except Exception:
        logging.exception("Failed to initialize GeminiVision; continuing without CV.")
        cv = None

    intents = discord.Intents.default()
    intents.guilds = True
    intents.message_content = True
    intents.messages = True

    client = LioHarness(
        channel_id=channel_id,
        channel_name=channel_name,
        chatter=chatter,
        store=store,
        overhead=overhead,
        tenor=tenor,
        user_memory=user_memory,
        daily_post=daily_post,
        cv=cv,
        intents=intents
    )
    client.run(token)

if __name__ == "__main__":
    main()
