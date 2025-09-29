# harness.py — autostart local vision + VLM-on-media, projector & chat_format auto-retry
import os
import re
import sys
import time
import atexit
import signal
import logging
import asyncio
import mimetypes
import subprocess
import base64
import io
from datetime import datetime
from glob import glob
from threading import Thread
from typing import Optional, List, Tuple

import requests
import discord
from dotenv import load_dotenv

from chat_store import ConversationStore
from chatter import Chatter, ChatMessage, build_provider
from overhead import Overhead, OverheadDecision
from tenor_client import TenorClient
from user_memory import UserMemoryStore, MemoryConfig
from vision import build_vision_engine, VisionEngine, VisionResult  # keep your vision engine

try:
    from zoneinfo import ZoneInfo
except ImportError:
    raise SystemExit("Python 3.9+ with zoneinfo is required; use Python 3.10+ for best results.")

try:
    from PIL import Image
except Exception:
    Image = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ----------------- VLM autostart with projector + chat_format auto-retry -----------------
_VLM_PROC: Optional[subprocess.Popen] = None
_VLM_OWNED = False
_VLM_LOG_PATH: Optional[str] = None

def _api_base() -> str:
    return (os.getenv("VLM_API_BASE") or "http://127.0.0.1:5000/v1").rstrip("/")

def _is_server_alive() -> bool:
    try:
        r = requests.get(_api_base() + "/models", timeout=1.5)
        return r.ok
    except Exception:
        return False

def _tail_file(path: Optional[str], max_lines: int = 120) -> str:
    if not path or not os.path.exists(path):
        return "(log file missing)"
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            chunk = 8192
            data = bytearray()
            lines = 0
            while size > 0 and lines <= max_lines:
                step = min(chunk, size)
                size -= step
                f.seek(size)
                data[:0] = f.read(step)
                lines = data.count(b"\n")
        text = data.decode("utf-8", errors="replace")
        return "\n".join(text.splitlines()[-max_lines:])
    except Exception as e:
        return f"(failed to read log: {e})"

def _stop_vlm():
    global _VLM_PROC, _VLM_OWNED
    if _VLM_PROC and _VLM_OWNED:
        try:
            logging.info("Stopping local VLM server...")
            if os.name == "nt":
                _VLM_PROC.terminate()
            else:
                _VLM_PROC.send_signal(signal.SIGINT)
            try:
                _VLM_PROC.wait(timeout=8)
            except Exception:
                _VLM_PROC.kill()
        except Exception:
            pass
    _VLM_PROC = None

def _auto_detect_paths() -> Tuple[Optional[str], Optional[str]]:
    here = os.path.dirname(os.path.abspath(__file__))
    ggufs = sorted(glob(os.path.join(here, "*.gguf")), key=lambda p: os.path.getsize(p))
    if not ggufs:
        return None, None

    mmproj = None
    for p in ggufs:
        name = os.path.basename(p).lower()
        if "mmproj" in name or "proj" in name:
            mmproj = p
            break

    model = ggufs[-1]
    try:
        same = os.path.samefile(model, mmproj) if mmproj else False
    except Exception:
        same = (model == (mmproj or ""))
    if mmproj and same and len(ggufs) >= 2:
        model = ggufs[-2]

    if not mmproj and len(ggufs) >= 2:
        mmproj = ggufs[0]

    try:
        if mmproj and os.path.samefile(model, mmproj):
            mmproj = None
    except Exception:
        if mmproj and (model == mmproj):
            mmproj = None
    return model, mmproj

def _launch_server(cmd: List[str], log_path: str) -> Optional[subprocess.Popen]:
    creationflags = 0
    if os.name == "nt":
        try:
            creationflags = subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]
        except Exception:
            creationflags = 0
    try:
        logf = open(log_path, "ab", buffering=0)
    except Exception:
        logf = open(os.devnull, "wb")
    try:
        return subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, creationflags=creationflags)
    except Exception as e:
        logging.error("Failed to start llama_cpp.server: %s", e)
        return None

def _background_wait_until_ready(proc: subprocess.Popen, log_path: str, max_wait_s: int = 45):
    t0 = time.time()
    while time.time() - t0 < max_wait_s:
        if proc.poll() is not None:
            code = proc.returncode
            logging.error("VLM server exited early with code %s. See %s for details.", code, log_path)
            tail = _tail_file(log_path)
            logging.error("---- vlm_server.log (last lines) ----\n%s\n---- end log ----", tail)
            return
        if _is_server_alive():
            logging.info("VLM server is ready at %s", _api_base())
            return
        time.sleep(1)

    logging.warning("VLM server not ready after %ss; continuing anyway. Check %s for progress/errors.",
                    max_wait_s, log_path)
    tail = _tail_file(log_path)
    logging.warning("---- vlm_server.log (last lines) ----\n%s\n---- end log ----", tail)

def ensure_vlm_server():
    """
    Start llama_cpp.server if VLM_AUTOSTART=1.
    Tries projector flag (--mmproj, then --clip_model_path) and common multimodal chat formats
    (--chat_format llava-1-6, llava-1-5), then falls back gracefully.
    """
    global _VLM_PROC, _VLM_OWNED, _VLM_LOG_PATH

    if os.getenv("VLM_AUTOSTART", "0") != "1":
        logging.info("VLM_AUTOSTART is off; skipping autostart.")
        return
    if _is_server_alive():
        logging.info("VLM server already running at %s", _api_base())
        return

    env_model  = os.getenv("VLM_MODEL_PATH")
    env_mmproj = os.getenv("VLM_MMPROJ_PATH")
    auto_model, auto_mmproj = _auto_detect_paths()

    model_path  = env_model  if (env_model  and os.path.exists(env_model))  else auto_model
    mmproj_path = env_mmproj if (env_mmproj and os.path.exists(env_mmproj)) else auto_mmproj

    if not model_path:
        logging.warning("No model .gguf found. Put your *.gguf files next to harness.py or set VLM_MODEL_PATH in .env.")
        return

    host = os.getenv("VLM_HOST", "127.0.0.1")
    port = os.getenv("VLM_PORT", "5000")
    n_gpu = os.getenv("VLM_N_GPU_LAYERS", "0")
    extra = os.getenv("VLM_SERVER_ARGS", "").strip()
    extra_args = extra.split() if extra else []

    here = os.path.dirname(os.path.abspath(__file__))
    log_path = os.path.join(here, "vlm_server.log")
    _VLM_LOG_PATH = log_path

    chat_formats = [s.strip() for s in (os.getenv("VLM_CHAT_FORMATS") or "llava-1-6,llava-1-5,llava").split(",") if s.strip()]

    def _base_cmd() -> List[str]:
        return [sys.executable, "-m", "llama_cpp.server",
                "--model", model_path,
                "--host", host, "--port", str(port),
                "--n_gpu_layers", str(n_gpu)] + (extra_args or [])

    attempts: List[Tuple[str, List[str]]] = []

    if mmproj_path and os.path.exists(mmproj_path):
        # projector + chat formats
        for cf in chat_formats:
            attempts.append((f"mmproj+cf={cf}", _base_cmd() + ["--mmproj", mmproj_path, "--chat_format", cf]))
        for cf in chat_formats:
            attempts.append((f"clip+cf={cf}", _base_cmd() + ["--clip_model_path", mmproj_path, "--chat_format", cf]))
        # projector without forcing chat format
        attempts.append(("mmproj", _base_cmd() + ["--mmproj", mmproj_path]))
        attempts.append(("clip",   _base_cmd() + ["--clip_model_path", mmproj_path]))

    # last resort: no projector (text-only)
    attempts.append(("none", _base_cmd()))

    for mode, cmd in attempts:
        logging.info("Starting local VLM server with:\n  model = %s\n  mmproj= %s\n  host  = %s\n  port  = %s",
                     model_path, (mmproj_path if mode != "none" else "(none)"), host, port)
        if "none" in mode and mmproj_path:
            logging.warning("Launching WITHOUT projector (last resort). Images may not work.")

        proc = _launch_server(cmd, log_path)
        if not proc:
            continue

        # quick gate: detect immediate failures & retry next attempt
        ok = False
        t0 = time.time()
        while time.time() - t0 < 8:
            if proc.poll() is not None:
                tail = _tail_file(log_path)
                logging.error("---- vlm_server.log (last lines) ----\n%s\n---- end log ----", tail)
                break
            if _is_server_alive():
                _VLM_PROC = proc
                _VLM_OWNED = True
                atexit.register(_stop_vlm)
                Thread(target=_background_wait_until_ready, args=(proc, log_path, 45), daemon=True).start()
                ok = True
                break
            time.sleep(0.5)
        if ok:
            return
        try:
            proc.terminate()
        except Exception:
            pass

# ----------------- Bot plumbing (unchanged behavior) -----------------

def now_eastern() -> datetime:
    tz_name = os.getenv("TZ", "America/New_York")
    try:
        tz = ZoneInfo(tz_name)
    except Exception:
        logging.warning("Invalid TZ '%s'; falling back to America/New_York", tz_name)
        tz = ZoneInfo("America/New_York")
    return datetime.now(tz=tz)

class LioHarness(discord.Client):
    def __init__(self, *, channel_id: int | None, channel_name: str,
                 chatter: Chatter, store: ConversationStore,
                 overhead: Overhead, tenor: TenorClient,
                 user_memory: UserMemoryStore,
                 vision: VisionEngine,
                 **kwargs):
        super().__init__(**kwargs)
        self.channel_id = channel_id
        self.channel_name = channel_name
        self.chatter = chatter
        self.store = store
        self.overhead = overhead
        self.tenor = tenor
        self.user_memory = user_memory
        self.vision = vision
        self._announced = False

        self.gif_random_prob = float(os.getenv("GIF_RANDOM_PROB", "0.15"))
        self.force_helper_on_media = os.getenv("VISION_FORCE_HELPER_ON_MEDIA", "1") == "1"
        self.max_ocr_chars_in_prompt = int(os.getenv("VISION_MAX_OCR_CHARS", "500"))
        self.vlm_timeout_s = max(int(os.getenv("VLM_CHAT_TIMEOUT_S", "90")), 60)  # floor 60s on CPU
        self.use_vlm_on_media = os.getenv("VLM_ON_MEDIA", "1") == "1"

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
        if message.author == self.user:
            return
        if not await self._is_target_channel(message.channel):
            return

        content = message.content or ""
        author_name = getattr(message.author, "display_name", message.author.name)

        try:
            self.user_memory.note_user_text(message.author.id, author_name, content)
        except Exception:
            logging.exception("note_user_text failed")

        self.store.add_user_message(
            channel_id=message.channel.id,
            author_id=message.author.id,
            author_name=author_name,
            content=content
        )

        is_mention = (self.user in getattr(message, "mentions", []))
        is_reply_to_bot = False
        if message.reference and message.reference.resolved:
            try:
                is_reply_to_bot = getattr(message.reference.resolved, "author", None).id == self.user.id
            except Exception:
                is_reply_to_bot = False
        if not (is_mention or is_reply_to_bot):
            return

        attachments = await self._collect_image_attachments(message)

        # ---- Try local VLM for media messages ----
        if attachments and self.use_vlm_on_media:
            try:
                vlm_text = await asyncio.to_thread(self._query_local_vlm_answer, attachments, content)
            except Exception:
                vlm_text = None
                logging.exception("Local VLM query failed unexpectedly.")
            if vlm_text:
                decision: OverheadDecision = self.overhead.decide(
                    channel_id=message.channel.id,
                    user_name=author_name,
                    user_text=content,
                    transcript_excerpt="(media message)"
                )
                if decision.emoji:
                    vlm_text = f"{vlm_text} {decision.emoji}"
                try:
                    await message.channel.send(vlm_text)
                    self.store.add_assistant_message(channel_id=message.channel.id, content=vlm_text)
                except Exception:
                    logging.exception("Failed to send VLM reply.")
                return
            else:
                if _VLM_LOG_PATH:
                    tail = _tail_file(_VLM_LOG_PATH)
                    logging.error("Local VLM unreachable/empty; falling back to GPT.\n---- vlm_server.log (last lines) ----\n%s\n---- end log ----", tail)
                else:
                    logging.error("Local VLM unreachable/empty; falling back to GPT.")

        # ---- Helper vision + GPT fallback ----
        vision_results: list[VisionResult] = []
        if attachments:
            for att in attachments:
                try:
                    res: VisionResult = await asyncio.to_thread(
                        self.vision.analyze_bytes,
                        att["bytes"],
                        mime_type=att["mime"],
                        filename=att["name"],
                    )
                    vision_results.append(res)
                    self.store.add_user_message(
                        channel_id=message.channel.id,
                        author_id=message.author.id,
                        author_name=author_name,
                        content=f"[media attached: {att['mime']}]"
                    )
                except Exception:
                    logging.exception("Vision analysis failed for %s", att["name"])

        history = self.store.export_chat_messages(channel_id=message.channel.id, include_author_names=True)
        excerpt = "\n".join(m.content for m in history)[-1000:]

        decision: OverheadDecision = self.overhead.decide(
            channel_id=message.channel.id,
            user_name=author_name,
            user_text=content,
            transcript_excerpt=excerpt,
        )

        try:
            if (decision.route != "GIF") and not vision_results:
                import random
                if random.random() < self.gif_random_prob:
                    decision = OverheadDecision(route="GIF", emoji=decision.emoji, gif_query=self._derive_gif_query(content), raw_command="RANDOM_GIF")
        except Exception:
            logging.exception("Random GIF overlay failed.")

        if vision_results and self.force_helper_on_media and decision.route != "C":
            decision = OverheadDecision(route="C", emoji=decision.emoji, gif_query=None, raw_command=decision.raw_command)

        if decision.route == "GIF" and not vision_results:
            await self._send_gif_reply(message.channel, decision)
            return

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

        reply_text = self._strip_emojis(reply_text) or "…"
        if decision.emoji:
            reply_text = f"{reply_text} {decision.emoji}"

        try:
            await message.channel.send(reply_text)
            self.store.add_assistant_message(channel_id=message.channel.id, content=reply_text)
        except Exception:
            logging.exception("Failed to send persona reply.")

    # ---------- Local VLM call ----------
    def _server_model_id(self, requested: Optional[str]) -> Optional[str]:
        try:
            r = requests.get(_api_base() + "/models", timeout=2.5)
            if not r.ok:
                return requested
            data = r.json() or {}
            items = (data.get("data") or []) if isinstance(data, dict) else []
            ids = [it.get("id") for it in items if isinstance(it, dict) and it.get("id")]
            if not ids:
                return requested
            if requested and requested in ids:
                return requested
            return ids[0]
        except Exception:
            return requested

    def _query_local_vlm_answer(self, attachments: list[dict], user_text: str) -> str | None:
        if not _is_server_alive():
            return None
        base = _api_base()
        model = self._server_model_id(os.getenv("VLM_MODEL")) or "default"
        timeout_s = self.vlm_timeout_s

        parts = [{"type": "text", "text": (user_text.strip() or "Please describe the image and answer concisely.")}]
        for att in attachments[:3]:
            mime = att.get("mime") or "image/png"
            data = att.get("bytes") or b""
            name = (att.get("name") or "").lower()
            try:
                if (mime == "image/gif" or name.endswith(".gif")) and Image is not None:
                    im = Image.open(io.BytesIO(data)); im.seek(0)
                    buf = io.BytesIO(); im.convert("RGB").save(buf, format="PNG")
                    data = buf.getvalue(); mime = "image/png"
            except Exception:
                pass
            b64 = base64.b64encode(data).decode("ascii")
            url = f"data:{mime};base64,{b64}"
            parts.append({"type": "image_url", "image_url": {"url": url}})

        payload = {"model": model, "messages": [{"role": "user", "content": parts}],
                   "temperature": float(os.getenv("VLM_TEMPERATURE", "0.2")),
                   "max_tokens": int(os.getenv("VLM_MAX_TOKENS", "256"))}

        try:
            r = requests.post(base + "/chat/completions", json=payload, timeout=timeout_s)
            if r.ok:
                data = r.json()
                msg = data.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
                msg = (msg or "").strip()
                if msg:
                    return msg
            # Older servers: prompt + images[]
            images_b64 = [p["image_url"]["url"].split(",", 1)[1] for p in parts if p.get("type") == "image_url"]
            fallback = {"model": model, "prompt": parts[0]["text"], "images": images_b64,
                        "temperature": float(os.getenv("VLM_TEMPERATURE", "0.2")),
                        "max_tokens": int(os.getenv("VLM_MAX_TOKENS", "256"))}
            r2 = requests.post(base + "/chat/completions", json=fallback, timeout=timeout_s)
            if r2.ok:
                data = r2.json()
                msg = data.get("choices", [{}])[0].get("message", {}).get("content", "") \
                      or data.get("choices", [{}])[0].get("text", "")
                msg = (msg or "").strip()
                if msg:
                    return msg
        except Exception as e:
            logging.error("Local VLM call threw: %s", e)
        return None

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

    async def _collect_image_attachments(self, message: discord.Message) -> list[dict]:
        out: list[dict] = []
        for a in getattr(message, "attachments", []) or []:
            mime = a.content_type or (mimetypes.guess_type(a.filename)[0] or "")
            if not (mime.startswith("image/")): continue
            try:
                data = await a.read()
                out.append({"name": a.filename, "mime": mime, "bytes": data})
            except Exception:
                logging.exception("Failed to read attachment %s", a.filename)
        return out

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

def main():
    load_dotenv()
    ensure_vlm_server()

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
    vision = build_vision_engine()

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
        vision=vision,
        intents=intents
    )
    client.run(token)

if __name__ == "__main__":
    main()
