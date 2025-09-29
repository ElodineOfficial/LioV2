# vision.py
# --------------------------------------------------------------------
# Pluggable CV engine for your Discord bot.
# Providers:
#   - lens        : Google Lens (web UI) via Puppeteer (Node) -- no OAI
#   - google      : Google Cloud Vision (official APIs)
#   - local       : local heuristics + optional Tesseract OCR
#   - lens_hybrid : Lens for entity ID + (optional) OCR from Tesseract
#
# Default (auto): prefer `lens` if LENS_WORKER_PATH is present,
#                 else prefer `google` if GCV creds are present,
#                 else fallback to `local`.
#
# Important: We **never** surface page titles/EXIF titles to the LLM.
#            We also strip EXIF by default before analysis.
# --------------------------------------------------------------------

from __future__ import annotations

import io
import os
import json
import logging
import tempfile
import subprocess
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, Literal, Any

from PIL import Image, ImageSequence, UnidentifiedImageError  # type: ignore

# Optional dependencies we soft-import
try:
    from google.cloud import vision as gcv  # type: ignore
    _HAS_GCV = True
except Exception:
    _HAS_GCV = False

try:
    import pytesseract  # type: ignore
    _HAS_TESS = True
except Exception:
    _HAS_TESS = False


LikelihoodMap = {
    0: "UNKNOWN",
    1: "VERY_UNLIKELY",
    2: "UNLIKELY",
    3: "POSSIBLE",
    4: "LIKELY",
    5: "VERY_LIKELY",
}

IdentKind = Literal["web", "object", "logo", "landmark", "label"]


@dataclass
class Identification:
    name: str
    kind: IdentKind
    score: float = 0.0
    bbox: Optional[List[float]] = None
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VisionConfig:
    # caps & toggles
    labels_topk: int = int(os.getenv("VISION_LABELS_TOPK", "8"))
    web_topk: int = int(os.getenv("VISION_WEB_TOPK", "8"))
    objects_topk: int = int(os.getenv("VISION_OBJECTS_TOPK", "10"))
    gif_max_frames: int = int(os.getenv("VISION_GIF_MAX_FRAMES", "4"))
    enable_labels: bool = os.getenv("VISION_ENABLE_LABELS", "1") == "1"
    enable_objects: bool = os.getenv("VISION_ENABLE_OBJECTS", "1") == "1"
    enable_logos: bool = os.getenv("VISION_ENABLE_LOGOS", "1") == "1"
    enable_landmarks: bool = os.getenv("VISION_ENABLE_LANDMARKS", "1") == "1"
    enable_web: bool = os.getenv("VISION_ENABLE_WEB", "1") == "1"
    enable_ocr: bool = os.getenv("VISION_ENABLE_OCR", "1") == "1"
    enable_safe: bool = os.getenv("VISION_ENABLE_SAFE", "1") == "1"
    # hygiene
    exclude_titles: bool = os.getenv("VISION_EXCLUDE_TITLES", "1") == "1"
    strip_exif: bool = os.getenv("VISION_STRIP_EXIF", "1") == "1"
    image_max_dim: int = int(os.getenv("VISION_IMAGE_MAX_DIM", "0"))
    # puppeteer (Lens)
    lens_worker_path: Optional[str] = os.getenv("LENS_WORKER_PATH") or None
    node_bin: str = os.getenv("NODE_BIN", "node")
    lens_headless: bool = os.getenv("LENS_HEADLESS", "1") != "0"
    lens_nav_timeout_ms: int = int(os.getenv("LENS_NAV_TIMEOUT_MS", "25000"))
    lens_delay_ms: int = int(os.getenv("LENS_DELAY_MS", "600"))
    lens_max_entities: int = int(os.getenv("LENS_MAX_ENTITIES", "12"))


@dataclass
class VisionResult:
    media_type: Literal["image", "gif"]
    identifications: List[Identification]
    ocr_text: str
    safety: Dict[str, str]               # adult/violence/racy/medical/spoof
    frames_sampled: int
    warnings: List[str]


class VisionProvider(Protocol):
    def analyze_image_bytes(self, content: bytes) -> VisionResult: ...


# -------------------------------
# Helpers
# -------------------------------
_TITLE_SEPS = (" - ", " | ", " • ", " — ", " – ")

def _is_titleish(s: str) -> bool:
    if not s:
        return False
    t = s.strip()
    if len(t) > 80:                   # long SEO titles out
        return True
    low = t.lower()
    if any(sep in t for sep in _TITLE_SEPS):
        return True
    if low.startswith(("screenshot", "img_", "img-", "file:", "untitled", "image titled")):
        return True
    if any(b in low for b in ("imgur", "pinterest", "tumblr", "reddit", "twitter", "facebook", "tenor", "gfycat")):
        return True
    return False


def _resize_and_strip_exif(content: bytes, *, max_dim: int, strip_exif: bool) -> bytes:
    if not (strip_exif or (max_dim and max_dim > 0)):
        return content
    try:
        im = Image.open(io.BytesIO(content)).convert("RGB")
        if max_dim and max_dim > 0:
            w, h = im.size
            m = max(w, h)
            if m > max_dim:
                scale = max_dim / float(m)
                im = im.resize((int(w * scale), int(h * scale)))
        out = io.BytesIO()
        im.save(out, format="PNG")  # re-encode strips EXIF
        return out.getvalue()
    except Exception:
        return content


def _norm_bbox(vertices) -> Optional[List[float]]:
    try:
        xs = [max(0.0, min(1.0, float(v.x))) for v in vertices]
        ys = [max(0.0, min(1.0, float(v.y))) for v in vertices]
        return [min(xs), min(ys), max(xs), max(ys)]
    except Exception:
        return None


def _merge_results(results: List[VisionResult], *, labels_topk: int, web_topk: int, objects_topk: int) -> VisionResult:
    agg: Dict[tuple, Identification] = {}
    ocr_text = ""
    safety: Dict[str, str] = {}
    warnings: List[str] = []
    frames = 0

    # dedupe by (kind, lower(name))
    for r in results:
        frames += r.frames_sampled
        warnings.extend(r.warnings or [])
        if r.ocr_text and len(ocr_text) < 5000:
            ocr_text += ("\n" if ocr_text else "") + r.ocr_text.strip()
        for k, v in (r.safety or {}).items():
            cur = safety.get(k)
            if cur is None:
                safety[k] = v
            else:
                order = ["UNKNOWN", "VERY_UNLIKELY", "UNLIKELY", "POSSIBLE", "LIKELY", "VERY_LIKELY"]
                if order.index(v) > order.index(cur):
                    safety[k] = v
        for it in r.identifications:
            key = (it.kind, it.name.lower())
            prev = agg.get(key)
            if (prev is None) or (it.score > prev.score):
                agg[key] = it

    # preference ordering
    kind_rank = {"landmark": 5, "logo": 4, "object": 3, "web": 2, "label": 1}
    merged = sorted(agg.values(), key=lambda x: (kind_rank.get(x.kind, 0), x.score), reverse=True)

    # cap by kind
    objs = [i for i in merged if i.kind == "object"][:objects_topk]
    web = [i for i in merged if i.kind == "web"][:web_topk]
    other = [i for i in merged if i.kind not in ("object", "web")]
    labels = [i for i in other if i.kind == "label"][:labels_topk]
    tops = [i for i in other if i.kind in ("landmark", "logo")]

    return VisionResult(
        media_type="image",
        identifications=tops + objs + web + labels,
        ocr_text=ocr_text.strip(),
        safety=safety,
        frames_sampled=max(1, frames),
        warnings=warnings,
    )


# -------------------------------
# Provider: Google Cloud Vision
# -------------------------------
class GoogleVisionProvider:
    def __init__(self, *, config: Optional[VisionConfig] = None):
        if not _HAS_GCV:
            raise RuntimeError("google-cloud-vision not installed. pip install google-cloud-vision")
        self.client = gcv.ImageAnnotatorClient()
        self.config = config or VisionConfig()

    def analyze_image_bytes(self, content: bytes) -> VisionResult:
        cfg = self.config
        content = _resize_and_strip_exif(content, max_dim=cfg.image_max_dim, strip_exif=cfg.strip_exif)
        image = gcv.Image(content=content)

        idents: List[Identification] = []
        ocr_text = ""
        safety: Dict[str, str] = {}
        warnings: List[str] = []

        if cfg.enable_objects:
            try:
                r = self.client.object_localization(image=image)
                for o in (r.localized_object_annotations or []):
                    if not o.name:
                        continue
                    idents.append(Identification(
                        name=o.name.strip(),
                        kind="object",
                        score=float(getattr(o, "score", 0.0) or 0.0),
                        bbox=_norm_bbox(o.bounding_poly.normalized_vertices)
                    ))
            except Exception as e:
                warnings.append(f"object_localization failed: {e}")

        if cfg.enable_logos:
            try:
                r = self.client.logo_detection(image=image)
                for lg in (r.logo_annotations or []):
                    if lg.description:
                        idents.append(Identification(
                            name=lg.description.strip(),
                            kind="logo",
                            score=float(getattr(lg, "score", 0.0) or 0.0),
                            bbox=_norm_bbox(getattr(lg, "bounding_poly", {}).normalized_vertices)
                            if getattr(lg, "bounding_poly", None) else None
                        ))
            except Exception as e:
                warnings.append(f"logo_detection failed: {e}")

        if cfg.enable_landmarks:
            try:
                r = self.client.landmark_detection(image=image)
                for lm in (r.landmark_annotations or []):
                    if lm.description:
                        idents.append(Identification(
                            name=lm.description.strip(),
                            kind="landmark",
                            score=float(getattr(lm, "score", 0.0) or 0.0),
                            bbox=_norm_bbox(getattr(lm, "bounding_poly", {}).normalized_vertices)
                            if getattr(lm, "bounding_poly", None) else None
                        ))
            except Exception as e:
                warnings.append(f"landmark_detection failed: {e}")

        if cfg.enable_labels:
            try:
                r = self.client.label_detection(image=image)
                for lb in (r.label_annotations or [])[: cfg.labels_topk]:
                    if lb.description:
                        idents.append(Identification(
                            name=lb.description.strip(),
                            kind="label",
                            score=float(getattr(lb, "score", 0.0) or 0.0),
                        ))
            except Exception as e:
                warnings.append(f"label_detection failed: {e}")

        if cfg.enable_web:
            try:
                r = self.client.web_detection(image=image)
                wd = getattr(r, "web_detection", None)
                if wd and getattr(wd, "web_entities", None):
                    for we in wd.web_entities:
                        desc = (getattr(we, "description", "") or "").strip()
                        if not desc:
                            continue
                        if cfg.exclude_titles and _is_titleish(desc):
                            continue
                        idents.append(Identification(
                            name=desc,
                            kind="web",
                            score=float(getattr(we, "score", 0.0) or 0.0)
                        ))
                # NOTE: do NOT read page/best-guess titles
            except Exception as e:
                warnings.append(f"web_detection failed: {e}")

        if cfg.enable_ocr:
            try:
                tr = self.client.text_detection(image=image)
                if tr and tr.full_text_annotation and getattr(tr.full_text_annotation, "text", None):
                    ocr_text = tr.full_text_annotation.text.strip()
                elif tr and tr.text_annotations:
                    ocr_text = (tr.text_annotations[0].description or "").strip()
            except Exception as e:
                warnings.append(f"text_detection failed: {e}")

        if cfg.enable_safe:
            try:
                sr = self.client.safe_search_detection(image=image)
                if sr and sr.safe_search_annotation:
                    ss = sr.safe_search_annotation
                    safety = {
                        "adult":    LikelihoodMap.get(int(ss.adult), "UNKNOWN"),
                        "violence": LikelihoodMap.get(int(ss.violence), "UNKNOWN"),
                        "racy":     LikelihoodMap.get(int(ss.racy), "UNKNOWN"),
                        "medical":  LikelihoodMap.get(int(ss.medical), "UNKNOWN"),
                        "spoof":    LikelihoodMap.get(int(ss.spoof), "UNKNOWN"),
                    }
            except Exception as e:
                warnings.append(f"safe_search_detection failed: {e}")

        # rank/dedupe/cap
        return _merge_results(
            [VisionResult("image", idents, ocr_text, safety, 1, warnings)],
            labels_topk=cfg.labels_topk, web_topk=cfg.web_topk, objects_topk=cfg.objects_topk
        )


# ------------------------------------
# Provider: Local-lite (no web calls)
# ------------------------------------
class LocalLiteProvider:
    def __init__(self, *, config: Optional[VisionConfig] = None):
        self.config = config or VisionConfig()

    def _ocr(self, content: bytes) -> str:
        if not (self.config.enable_ocr and _HAS_TESS):
            return ""
        try:
            img = Image.open(io.BytesIO(content))
            return (pytesseract.image_to_string(img) or "").strip()
        except Exception:
            return ""

    def analyze_image_bytes(self, content: bytes) -> VisionResult:
        idents: List[Identification] = []
        # crude color tags to give *something* useful; never titles
        try:
            img = Image.open(io.BytesIO(content)).convert("RGB")
            small = img.resize((64, 64))
            colors = small.getcolors(maxcolors=64 * 64) or []
            colors.sort(key=lambda t: t[0], reverse=True)
            if colors:
                r, g, b = colors[0][1]
                if r > 180 and g < 100 and b < 100:
                    idents.append(Identification("red-dominant", "label", 0.6))
                elif g > 150 and r < 120:
                    idents.append(Identification("green-dominant", "label", 0.6))
                elif b > 150 and r < 120:
                    idents.append(Identification("blue-dominant", "label", 0.6))
        except Exception:
            pass
        text = self._ocr(content)
        return VisionResult("image", idents[: self.config.labels_topk], text, {}, 1, [])


# ----------------------------------------------------
# Provider: Lens (web UI) via Puppeteer (Node runner)
# ----------------------------------------------------
class LensWebProvider:
    """
    Spawns a Node.js worker (Puppeteer) that uploads the image to https://lens.google.com
    and scrapes *entity-like* chips/labels. We aggressively filter any title-ish strings.
    """
    def __init__(self, *, config: Optional[VisionConfig] = None):
        self.config = config or VisionConfig()
        if not self.config.lens_worker_path:
            raise RuntimeError("LENS_WORKER_PATH is required for VISION_PROVIDER=lens")

    def _run_worker(self, tmp_path: str) -> Dict[str, Any]:
        env = os.environ.copy()
        env.setdefault("LENS_HEADLESS", "1" if self.config.lens_headless else "0")
        env.setdefault("LENS_NAV_TIMEOUT_MS", str(self.config.lens_nav_timeout_ms))
        env.setdefault("LENS_DELAY_MS", str(self.config.lens_delay_ms))
        env.setdefault("LENS_MAX_ENTITIES", str(self.config.lens_max_entities))
        try:
            cp = subprocess.run(
                [self.config.node_bin, self.config.lens_worker_path, "--file", tmp_path],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, check=False, text=True, timeout=60
            )
            if cp.returncode != 0:
                return {"entities": [], "warnings": [f"lens worker exit {cp.returncode}: {cp.stderr.strip()[:200]}"]}
            out = (cp.stdout or "").strip()
            data = json.loads(out) if out else {}
            if not isinstance(data, dict):
                return {"entities": [], "warnings": ["lens worker returned non-dict"]}
            return data
        except subprocess.TimeoutExpired:
            return {"entities": [], "warnings": ["lens worker timeout"]}
        except Exception as e:
            return {"entities": [], "warnings": [f"lens worker error: {e}"]}

    def analyze_image_bytes(self, content: bytes) -> VisionResult:
        cfg = self.config
        # Always strip EXIF before sending
        content = _resize_and_strip_exif(content, max_dim=cfg.image_max_dim, strip_exif=True)
        # Write to a temp file for Puppeteer
        with tempfile.NamedTemporaryFile(prefix="lens_", suffix=".png", delete=True) as tf:
            tf.write(content)
            tf.flush()
            data = self._run_worker(tf.name)

        warnings = list(data.get("warnings") or [])
        raw_entities: List[str] = list(data.get("entities") or [])
        idents: List[Identification] = []
        for s in raw_entities[: cfg.lens_max_entities]:
            s = (s or "").strip()
            if not s:
                continue
            if cfg.exclude_titles and _is_titleish(s):
                continue
            idents.append(Identification(name=s, kind="web", score=0.0))

        # OCR fallback with pytesseract if enabled
        ocr_text = ""
        if cfg.enable_ocr and _HAS_TESS:
            try:
                img = Image.open(io.BytesIO(content))
                ocr_text = (pytesseract.image_to_string(img) or "").strip()
            except Exception:
                pass

        return VisionResult(
            media_type="image",
            identifications=idents,
            ocr_text=ocr_text,
            safety={},                 # no SafeSearch from Lens UI
            frames_sampled=1,
            warnings=warnings,
        )


# ----------------------------------------------------
# Provider: Lens-hybrid (Lens + local OCR)
# ----------------------------------------------------
class LensHybridProvider:
    def __init__(self, *, config: Optional[VisionConfig] = None):
        self.config = config or VisionConfig()
        self.lens = LensWebProvider(config=self.config)
        # Hybrid keeps costs zero: we don't invoke GCV here; OCR via Tesseract only.

    def analyze_image_bytes(self, content: bytes) -> VisionResult:
        # Lens entities + local OCR already happen in LensWebProvider
        return self.lens.analyze_image_bytes(content)


# -------------------------------
# Orchestrator (handles GIFs)
# -------------------------------
# ... keep your imports and classes above as-is ...

class VisionEngine:
    def __init__(self, config: Optional[VisionConfig] = None):
        self.config = config or VisionConfig()

    def _select_provider(self) -> VisionProvider:
        choice = (os.getenv("VISION_PROVIDER") or "auto").lower().strip()
        has_lens = bool(self.config.lens_worker_path)
        gcv_ready = _HAS_GCV and (
            os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
            or os.getenv("GOOGLE_CLOUD_PROJECT")
            or os.getenv("GOOGLE_API_KEY")
        )

        if choice == "lens":
            return LensWebProvider(config=self.config)
        if choice == "lens_hybrid":
            return LensHybridProvider(config=self.config)
        if choice == "google":
            return GoogleVisionProvider(config=self.config)
        if choice in ("local", "llamacpp", "none"):
            # treat 'llamacpp' as an alias for local-lite helper
            return LocalLiteProvider(config=self.config)

        # auto
        if has_lens:
            return LensWebProvider(config=self.config)
        if gcv_ready:
            return GoogleVisionProvider(config=self.config)
        return LocalLiteProvider(config=self.config)


    def _sample_gif_frames(self, gif_bytes: bytes, max_frames: int) -> List[bytes]:
        frames: List[bytes] = []
        try:
            im = Image.open(io.BytesIO(gif_bytes))
        except UnidentifiedImageError:
            return frames
        total = getattr(im, "n_frames", 1)
        if total <= 1:  # static disguised as GIF
            buf = io.BytesIO()
            im.convert("RGB").save(buf, format="PNG")
            return [buf.getvalue()]
        take = max(1, min(max_frames, total))
        idxs = sorted(set(int(i * (total / take)) for i in range(take)))
        for idx in idxs:
            try:
                im.seek(idx)
                frm = im.convert("RGB")
                buf = io.BytesIO()
                frm.save(buf, format="PNG")  # strips EXIF
                frames.append(buf.getvalue())
            except Exception:
                continue
        return frames or [gif_bytes]

    def analyze_bytes(self, content: bytes, *, mime_type: str, filename: str) -> VisionResult:
        is_gif = (mime_type or "").lower() == "image/gif" or filename.lower().endswith(".gif")
        provider = self._select_provider()

        if not is_gif:
            return provider.analyze_image_bytes(content)

        # GIF: sample frames, run analysis per frame, merge
        sampled = self._sample_gif_frames(content, self.config.gif_max_frames)
        partials: List[VisionResult] = []
        warns: List[str] = []
        for idx, fb in enumerate(sampled):
            try:
                r = provider.analyze_image_bytes(fb)
                partials.append(r)
            except Exception as e:
                warns.append(f"frame {idx} analysis failed: {e}")

        merged = _merge_results(
            partials,
            labels_topk=self.config.labels_topk,
            web_topk=self.config.web_topk,
            objects_topk=self.config.objects_topk
        )
        merged.media_type = "gif"
        merged.frames_sampled = len(sampled)
        merged.warnings.extend(warns)
        return merged


def build_vision_engine() -> VisionEngine:
    return VisionEngine(config=VisionConfig())
