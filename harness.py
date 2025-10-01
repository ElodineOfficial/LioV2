# harness.py — use Discord typing indicator exclusively (no placeholder message)
import os
import re
import logging
from datetime import datetime

import discord
from dotenv import load_dotenv

from chat_store import ConversationStore
from chatter import Chatter, build_provider
from overhead import Overhead, OverheadDecision
from tenor_client import TenorClient
from user_memory import UserMemoryStore, MemoryConfig

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
                 **kwargs):
        super().__init__(**kwargs)
        self.channel_id = channel_id
        self.channel_name = channel_name
        self.chatter = chatter
        self.store = store
        self.overhead = overhead
        self.tenor = tenor
        self.user_memory = user_memory
        self._announced = False

        # Keep GIF behavior; no CV.
        self.gif_random_prob = float(os.getenv("GIF_RANDOM_PROB", "0.05"))

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

    async def on_message(self, message: discord.Message):
        # Ignore our own messages
        if message.author == self.user:
            return
        # Only respond in the configured channel (or its threads)
        if not await self._is_target_channel(message.channel):
            return

        content = message.content or ""
        author_name = getattr(message.author, "display_name", message.author.name)

        # Memory tap
        try:
            self.user_memory.note_user_text(message.author.id, author_name, content)
        except Exception:
            logging.exception("note_user_text failed")

        # Persist user message
        self.store.add_user_message(
            channel_id=message.channel.id,
            author_id=message.author.id,
            author_name=author_name,
            content=content
        )

        # Only respond when mentioned or replying to us
        is_mention = (self.user in getattr(message, "mentions", []))
        is_reply_to_bot = False
        if message.reference and message.reference.resolved:
            try:
                is_reply_to_bot = getattr(message.reference.resolved, "author", None).id == self.user.id
            except Exception:
                is_reply_to_bot = False
        if not (is_mention or is_reply_to_bot):
            return

        # ---------------------------------------------------------------
        # Show the native Discord typing indicator for ALL processing.
        # No 'thinking...' placeholder is posted to the chat.
        # ---------------------------------------------------------------
        decision: OverheadDecision | None = None
        reply_text: str | None = None

        try:
            async with message.channel.typing():
                # Build recent context AFTER we've begun typing
                history = self.store.export_chat_messages(
                    channel_id=message.channel.id, include_author_names=True
                )
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
                    await self._send_gif_reply(message.channel, decision)
                    return

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
            logging.exception("Unexpected failure in on_message processing.")
            reply_text = "My brain farted. Try again?"

        # Send final output (typing indicator stops automatically here)
        reply_text = self._strip_emojis(reply_text or "") or "…"
        if decision and decision.emoji:
            reply_text = f"{reply_text} {decision.emoji}"

        try:
            await message.channel.send(reply_text)
            self.store.add_assistant_message(channel_id=message.channel.id, content=reply_text)
        except Exception:
            logging.exception("Failed to send persona reply.")

    # ---------- helpers ----------

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

    async def _send_gif_reply(self, channel: discord.TextChannel, decision: OverheadDecision):
        url = await self.tenor.search_first_gif_url(decision.gif_query or "reaction")
        if url:
            try:
                sent = await channel.send(url)
                if decision.emoji:
                    from discord import PartialEmoji
                    try:
                        await sent.add_reaction(PartialEmoji.from_str(decision.emoji) if decision.emoji.startswith("<:") else decision.emoji)
                    except Exception:
                        pass
                self.store.add_assistant_message(channel_id=channel.id, content=f"[GIF] {url}")
                return
            except Exception:
                logging.exception("Failed sending GIF.")
        fallback = f"*{decision.gif_query or 'reaction'}*"
        try:
            sent = await channel.send(fallback)
            if decision.emoji:
                await sent.add_reaction(decision.emoji)
            self.store.add_assistant_message(channel_id=channel.id, content=fallback)
        except Exception:
            logging.exception("Failed sending GIF fallback.")

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
        intents=intents
    )
    client.run(token)

if __name__ == "__main__":
    main()
