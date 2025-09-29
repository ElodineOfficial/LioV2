# chatter.py
from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from typing import List, Protocol, Optional
import asyncio

# ---------- Message type (provider-agnostic) ----------

@dataclass
class ChatMessage:
    role: str  # "user" | "assistant"
    content: str


# ---------- Provider abstraction (keeps us vendor-agnostic) ----------

class LLMProvider(Protocol):
    def complete_chat(
        self,
        system_prompt: str,
        messages: List[ChatMessage],
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> str:
        """Return assistant text."""


# ---------- OpenAI provider (swappable) ----------

class OpenAIProvider:
    """
    Minimal OpenAI implementation behind the provider interface.
    Env:
      - OPENAI_API_KEY (required)
      - OPENAI_MODEL (default 'gpt-4o-mini')
      - OPENAI_BASE_URL (optional; proxies/compat)
    """
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.base_url = os.getenv("OPENAI_BASE_URL")

    def complete_chat(
        self,
        system_prompt: str,
        messages: List[ChatMessage],
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> str:
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is required or use LLM_PROVIDER=echo.")

        # Lazy import so the module loads even if openai isn't installed
        try:
            from openai import OpenAI  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "The 'openai' package is required. Install with: pip install openai"
            ) from e

        client = OpenAI(api_key=self.api_key, base_url=self.base_url or None)

        oai_msgs = []
        if system_prompt:
            oai_msgs.append({"role": "system", "content": system_prompt})
        for m in messages:
            role = "assistant" if m.role == "assistant" else "user"
            oai_msgs.append({"role": role, "content": m.content})

        resp = client.chat.completions.create(
            model=self.model,
            messages=oai_msgs,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return (resp.choices[0].message.content or "").strip()


# ---------- Dev/Test fallback provider ----------

class EchoProvider:
    """Safe fallback when no external key/provider is available."""
    def complete_chat(
        self,
        system_prompt: str,
        messages: List[ChatMessage],
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> str:
        last_user = next((m for m in reversed(messages) if m.role == "user"), None)
        sys_hint = " (sys present)" if system_prompt else ""
        return f"[echo{sys_hint}] {last_user.content if last_user else '…'}"


def build_provider() -> LLMProvider:
    """
    Choose provider via env:
      LLM_PROVIDER=openai|echo  (default: openai if key present else echo)
    """
    choice = (os.getenv("LLM_PROVIDER") or "").lower().strip()
    if choice == "openai":
        if os.getenv("OPENAI_API_KEY"):
            return OpenAIProvider()
        return EchoProvider()
    if choice == "echo":
        return EchoProvider()

    # Auto-detect
    if os.getenv("OPENAI_API_KEY"):
        return OpenAIProvider()
    return EchoProvider()


# ---------- Chatter orchestration ----------

class Chatter:
    """
    Orchestrates LLM calls:
      - Loads hidden instruction from a personality file by default (A)
      - Accepts provider-agnostic ChatMessage[]
      - Returns assistant text
    """
    def __init__(self, provider: LLMProvider, personality_path: str = "personalityA.txt"):
        self.provider = provider
        self.personality_path = personality_path
        self._system_prompt_cache: Optional[str] = None

    def _load_text_file(self, path: str) -> str:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read().strip()
        except FileNotFoundError:
            return ""

    def _load_system_prompt(self) -> str:
        if self._system_prompt_cache is not None:
            return self._system_prompt_cache
        txt = self._load_text_file(self.personality_path)
        if not txt:
            txt = (
                "You are Lio (Personality A): snarky, cheeky, but helpful. "
                "Keep responses concise unless asked otherwise."
            )
        self._system_prompt_cache = txt
        return txt

    def generate_reply(self, messages: List[ChatMessage]) -> str:
        system_prompt = self._load_system_prompt()
        return self.provider.complete_chat(system_prompt, messages)

    def generate_reply_with_system(self, messages: List[ChatMessage], system_prompt: str) -> str:
        if not system_prompt:
            system_prompt = self._load_system_prompt()
        return self.provider.complete_chat(system_prompt, messages)

    async def generate_reply_async(self, messages: List[ChatMessage]) -> str:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.generate_reply, messages)

    async def generate_reply_with_system_async(self, messages: List[ChatMessage], system_prompt: str) -> str:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.generate_reply_with_system, messages, system_prompt)
