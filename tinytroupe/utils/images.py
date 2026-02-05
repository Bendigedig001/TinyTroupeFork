import base64
import hashlib
import io
import json
import os
from dataclasses import dataclass
from typing import Optional

from tinytroupe.utils import logger

try:
    from PIL import Image, ImageOps
except Exception as exc:  # pragma: no cover - handled at runtime
    Image = None
    ImageOps = None
    _PIL_IMPORT_ERROR = exc
else:
    _PIL_IMPORT_ERROR = None


@dataclass(frozen=True)
class ImageSpec:
    path: str
    detail: str = "low"
    max_dim: int = 768
    format: str = "jpeg"
    quality: int = 85


@dataclass(frozen=True)
class ImageAsset:
    source_path: str
    cache_path: str
    mime_type: str
    width: int
    height: int
    size_bytes: int
    source_sha256: str
    cache_key: str
    format: str
    quality: int
    max_dim: int

    def data_url(self) -> str:
        data = _read_file_bytes(self.cache_path)
        encoded = base64.b64encode(data).decode("ascii")
        return f"data:{self.mime_type};base64,{encoded}"


def _require_pillow() -> None:
    if Image is None or ImageOps is None:
        raise RuntimeError(
            "Pillow is required for image preprocessing but is not installed."
        ) from _PIL_IMPORT_ERROR


def _read_file_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _normalize_format(fmt: Optional[str]) -> str:
    if not fmt:
        return "jpeg"
    fmt = fmt.strip().lower()
    if fmt == "jpg":
        return "jpeg"
    return fmt


def _format_to_ext(fmt: str) -> str:
    if fmt == "jpeg":
        return "jpg"
    return fmt


def _format_to_mime(fmt: str) -> str:
    if fmt == "jpeg":
        return "image/jpeg"
    if fmt == "png":
        return "image/png"
    if fmt == "webp":
        return "image/webp"
    return f"image/{fmt}"


def _build_cache_key(source_sha256: str, max_dim: int, fmt: str, quality: int) -> str:
    payload = json.dumps(
        {
            "v": 1,
            "source": source_sha256,
            "max_dim": int(max_dim),
            "format": fmt,
            "quality": int(quality),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _resize_dimensions(width: int, height: int, max_dim: int) -> tuple[int, int]:
    if max_dim <= 0:
        return width, height
    largest = max(width, height)
    if largest <= max_dim:
        return width, height
    scale = max_dim / float(largest)
    return max(1, int(round(width * scale))), max(1, int(round(height * scale)))


def preprocess_image_cached(spec: ImageSpec, cache_dir: str) -> ImageAsset:
    """
    Preprocesses an image (resize/convert/encode) and caches it on disk.
    Returns an ImageAsset pointing at the cached file.
    """
    _require_pillow()

    if not os.path.isfile(spec.path):
        raise FileNotFoundError(spec.path)

    fmt = _normalize_format(spec.format)
    if fmt not in {"jpeg", "png", "webp"}:
        raise ValueError(f"Unsupported image format: {fmt}")

    source_bytes = _read_file_bytes(spec.path)
    source_sha256 = _sha256_bytes(source_bytes)
    cache_key = _build_cache_key(source_sha256, spec.max_dim, fmt, spec.quality)

    images_dir = os.path.join(cache_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    ext = _format_to_ext(fmt)
    cache_path = os.path.join(images_dir, f"{cache_key}.{ext}")
    meta_path = os.path.join(images_dir, f"{cache_key}.json")

    if os.path.exists(cache_path) and os.path.exists(meta_path):
        try:
            meta = json.loads(_read_file_bytes(meta_path).decode("utf-8"))
            return ImageAsset(
                source_path=spec.path,
                cache_path=cache_path,
                mime_type=meta["mime_type"],
                width=int(meta["width"]),
                height=int(meta["height"]),
                size_bytes=int(meta["size_bytes"]),
                source_sha256=meta["source_sha256"],
                cache_key=cache_key,
                format=meta.get("format", fmt),
                quality=int(meta.get("quality", spec.quality)),
                max_dim=int(meta.get("max_dim", spec.max_dim)),
            )
        except Exception:
            logger.warning(
                "Image cache metadata invalid for %s, regenerating.", cache_path
            )

    with Image.open(io.BytesIO(source_bytes)) as img:
        img = ImageOps.exif_transpose(img)
        width, height = img.size
        new_width, new_height = _resize_dimensions(width, height, spec.max_dim)
        if (new_width, new_height) != (width, height):
            img = img.resize((new_width, new_height), Image.LANCZOS)

        if fmt == "jpeg":
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")
        elif fmt == "png":
            if img.mode == "P":
                img = img.convert("RGBA")

        out = io.BytesIO()
        save_params = {}
        if fmt in {"jpeg", "webp"}:
            save_params["quality"] = int(spec.quality)
            save_params["optimize"] = True
        img.save(out, format=fmt.upper(), **save_params)
        output_bytes = out.getvalue()

    with open(cache_path, "wb") as f:
        f.write(output_bytes)

    meta = {
        "source_sha256": source_sha256,
        "cache_key": cache_key,
        "format": fmt,
        "quality": int(spec.quality),
        "max_dim": int(spec.max_dim),
        "mime_type": _format_to_mime(fmt),
        "width": new_width,
        "height": new_height,
        "size_bytes": len(output_bytes),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return ImageAsset(
        source_path=spec.path,
        cache_path=cache_path,
        mime_type=meta["mime_type"],
        width=int(meta["width"]),
        height=int(meta["height"]),
        size_bytes=int(meta["size_bytes"]),
        source_sha256=source_sha256,
        cache_key=cache_key,
        format=fmt,
        quality=int(spec.quality),
        max_dim=int(spec.max_dim),
    )


def build_image_content_part(asset: ImageAsset, detail: str = "low") -> dict:
    """
    Builds an OpenAI-compatible image content part using a data URL.
    """
    detail_value = (detail or "low").strip().lower()
    if detail_value not in {"low", "high", "auto"}:
        detail_value = "low"
    return {
        "type": "image_url",
        "image_url": {"url": asset.data_url(), "detail": detail_value},
    }
