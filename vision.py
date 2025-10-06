# vision.py
from __future__ import annotations

import os
import json
import logging
from typing import List, Tuple, Optional

# Gemini API (google-genai / Gen AI SDK)
# pip install --upgrade google-genai
from google import genai
from google.genai import types


def _env_flag(name: str, default: bool) -> bool:
    raw = (os.getenv(name) or "").strip()
    if raw == "":
        return default
    return raw not in {"0", "false", "False", "no", "No"}


class GeminiVision:
    """
    Minimal image-caption/OCR summarizer for Discord attachments.
    - Reads bytes + mime for up to CV_MAX_IMAGES images
    - Asks Gemini to return a tiny JSON object
    - Returns a compact '[CV] {...}' string suitable to insert into chat history
    Env:
      - GEMINI_MODEL (default: 'gemini-2.5-flash-lite')
      - CV_MAX_IMAGES (default: 2)
    """

    def __init__(self, *, model: Optional[str] = None, logger: Optional[logging.Logger] = None):
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
        self.logger = logger or logging.getLogger(__name__)

        # The client automatically picks up GEMINI_API_KEY / GOOGLE_API_KEY from env.
        # https://ai.google.dev/gemini-api/docs/quickstart
        try:
            self.client = genai.Client()
        except Exception as e:
            raise RuntimeError(
                "Failed to initialize Gemini client. Ensure GEMINI_API_KEY is set."
            ) from e

        # Small knobs
        self.max_images = max(1, int(os.getenv("CV_MAX_IMAGES", "2")))

    def summarize(
        self,
        images: List[Tuple[bytes, str]],
        *,
        user_hint: str = "",
    ) -> Optional[str]:
        """
        :param images: list of (bytes, mime_type) for image/* attachments
        :param user_hint: short text from the user message (optional)
        :return: '[CV] {...}' compact JSON or None on failure/empty
        """
        if not images:
            return None

        # Build image parts (inline bytes). Inline total must be < ~20MB. See docs.
        # https://ai.google.dev/gemini-api/docs/image-understanding
        parts: List[types.Part] = []
        used = 0
        for data, mime in images:
            if used >= self.max_images:
                break
            mime = (mime or "").lower().strip() or "image/jpeg"
            try:
                parts.append(types.Part.from_bytes(data=data, mime_type=mime))
                used += 1
            except Exception:
                # Skip bad images but continue others
                self.logger.exception("Failed to convert image to Part (mime=%s)", mime)
                continue

        if not parts:
            return None

        instructions = (
            "You are a vision describer. Return ONLY JSON with these keys:\n"
            'caption: string (<=120 chars);\n'
            "objects: array of up to 8 short object names;\n"
            "text_in_image: string with any visible text (preserve line breaks; <=300 chars);\n"
            "people_count: integer (0 if none, -1 if unsure);\n"
            "safety_flags: array of zero or more from ['nsfw','violence','self-harm','drugs','weapons','graphic'];\n"
            "notable_details: array (<=5) of concise facts (e.g., logos, landmarks, UI elements).\n"
            "Be concise and factual. If text is unreadable, use an empty string for text_in_image."
        )
        if user_hint:
            instructions += f"\nUser hint: {user_hint[:200]}"

        try:
            # Force JSON output without adding extra deps/schemas.
            resp = self.client.models.generate_content(
                model=self.model,
                contents=parts + [instructions],
                # Structured output: JSON mime type (no schema).
                # https://ai.google.dev/gemini-api/docs/structured-output
                config={"response_mime_type": "application/json"},
            )
            text = (getattr(resp, "text", "") or "").strip()
            if not text:
                return None
            # Validate + compact the JSON
            data = json.loads(text)
            compact = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
            return f"[CV] {compact}"
        except Exception:
            self.logger.exception("Gemini vision summarization failed")
            return None
