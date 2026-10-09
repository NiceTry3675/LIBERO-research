"""Images for the decision model: zoomed object crops and tile sheets with large labels.

Clef reads large marks on zoomed crops well and small marks on full scenes badly, so every image made here is
local and its labels are large. Crops come from the raw top (straight down) and agent (oblique front) cameras,
centred on an object's projected footprint.
"""
from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .capture import Capture
from .geometry import project

FONT_PATHS = ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf")


def font(size: int):
    for path in FONT_PATHS:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def object_box(cap: Capture, cam: str, centre_xy, half_cm: float, z_low: float, z_high: float) -> tuple[int, int, int, int]:
    """Pixel box (full resolution) around the world box centre +- half_cm in x/y, from z_low to z_high (cm)."""
    K = cap.K(cam).copy()
    K[:2] *= 2 if cam in ("top_camera", "agent_camera") else 1      # stored K is for the halved depth
    cx, cy = centre_xy
    corners = np.array([[cx + dx, cy + dy, z] for dx in (-half_cm, half_cm) for dy in (-half_cm, half_cm)
                        for z in (z_low, z_high)])
    uv, _ = project(corners, K, cap.E(cam))
    u0, v0 = np.floor(uv.min(0)).astype(int)
    u1, v1 = np.ceil(uv.max(0)).astype(int)
    return int(u0), int(v0), int(u1), int(v1)


def crop(cap: Capture, cam: str, centre_xy, half_cm: float, height_cm: float, size: int = 256,
         label: str | None = None, outline: list | None = None, image: Image.Image | None = None) -> Image.Image:
    """A square crop of the raw camera image (or of `image`, the same view with marks drawn on it) around an
    object, resized to size x size, optionally labelled."""
    img = image if image is not None else Image.fromarray(cap.rgb(cam))
    u0, v0, u1, v1 = object_box(cap, cam, centre_xy, half_cm, cap.table_z, cap.table_z + max(height_cm, 1.0))
    side = max(u1 - u0, v1 - v0, 24)
    cu, cv = (u0 + u1) / 2, (v0 + v1) / 2
    box = (int(cu - side / 2), int(cv - side / 2), int(cu + side / 2), int(cv + side / 2))
    out = img.crop(box).resize((size, size), Image.LANCZOS)
    if label:
        d = ImageDraw.Draw(out)
        f = font(size // 6)
        tw = d.textlength(label, font=f)
        d.rectangle([0, 0, tw + 12, size // 6 + 10], fill=(0, 0, 0))
        d.text((6, 3), label, fill=(255, 255, 0), font=f)
    return out


def tile_sheet(tiles: list[Image.Image], cols: int = 3, gap: int = 6, background=(255, 255, 255)) -> Image.Image:
    """Tiles (already labelled) in a grid on one image."""
    if not tiles:
        return Image.new("RGB", (64, 64), background)
    w, h = tiles[0].size
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * w + (cols + 1) * gap, rows * h + (rows + 1) * gap), background)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        sheet.paste(t, (gap + c * (w + gap), gap + r * (h + gap)))
    return sheet


def jpeg_uri(img: Image.Image, quality: int = 90) -> str:
    import base64
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
