"""Create the upload pack and a bright thumbnail from the actual lesson artwork."""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from img2vid.render import _font

W, H = 1280, 720
NAVY = "#183C6C"
ROUNDED = Path("/System/Library/Fonts/Supplemental/Arial Rounded Bold.ttf")


def face(size):
    if ROUNDED.exists():
        return ImageFont.truetype(str(ROUNDED), size)
    return _font("ABC", size, None)


def star(image, center, radius, color, rotation=0):
    points = []
    for i in range(10):
        angle = math.radians(rotation - 90 + 36 * i)
        length = radius if i % 2 == 0 else radius * 0.45
        points.append((center[0] + math.cos(angle) * length,
                       center[1] + math.sin(angle) * length))
    ImageDraw.Draw(image).polygon(points, fill=color)


def sticker(image, name, crop, xy, diameter, border=10):
    """Preserve the real artwork inside a high-contrast circular picture sticker."""
    source = Image.open(PROJECT / "images" / name).convert("RGB").crop(crop)
    source = ImageOps.fit(source, (diameter, diameter), Image.Resampling.LANCZOS)
    x, y = xy
    shadow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(shadow).ellipse(
        (x - border + 3, y - border + 12, x + diameter + border + 3, y + diameter + border + 12),
        fill=(18, 64, 95, 90),
    )
    image.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(8)))
    ImageDraw.Draw(image).ellipse(
        (x - border, y - border, x + diameter + border, y + diameter + border),
        fill="white",
    )
    mask = Image.new("L", (diameter, diameter), 0)
    ImageDraw.Draw(mask).ellipse((0, 0, diameter - 1, diameter - 1), fill=255)
    source.putalpha(mask)
    image.alpha_composite(source, (x, y))


def chunky_letter(image, letter, center, fill):
    draw = ImageDraw.Draw(image)
    f = face(224)
    x, y = center
    # A short extrusion and a white edge keep each glyph readable on a phone.
    for depth in range(12, 0, -2):
        draw.text((x + 3, y + depth), letter, font=f, anchor="mm",
                  fill=NAVY, stroke_width=12, stroke_fill=NAVY)
    draw.text((x, y), letter, font=f, anchor="mm",
              fill=fill, stroke_width=9, stroke_fill="white")


def thumbnail():
    yy, xx = np.mgrid[0:H, 0:W]
    blend = np.clip(0.6 * yy / H + 0.4 * xx / W, 0, 1)[..., None]
    top = np.array([255, 244, 143])
    bottom = np.array([255, 205, 75])
    image = Image.fromarray((top * (1 - blend) + bottom * blend).astype("uint8")).convert("RGBA")
    draw = ImageDraw.Draw(image)
    draw.ellipse((665, -100, 1505, 830), fill="#2CC7D8")
    draw.ellipse((709, -70, 1459, 750), outline="#6EE1E5", width=14)
    draw.ellipse((-155, 510, 120, 785), fill="#FFB34C")
    for center, radius, color, rotation in [
        ((686, 157), 24, "white", 15),
        ((694, 385), 19, "#FF7D5A", 8),
        ((1235, 432), 25, "#FFF075", 18),
        ((70, 562), 14, "#FFA444", -10),
        ((620, 572), 15, "#FF865B", 22),
    ]:
        star(image, center, radius, color, rotation)

    # The kitten is the friendly focal point; the fruit/toys preview other lesson words.
    sticker(image, "03.png", (350, 40, 910, 600), (730, 42), 510, border=12)
    sticker(image, "01.png", (270, 30, 1000, 720), (687, 493), 193, border=9)
    sticker(image, "02.png", (260, 30, 1000, 720), (895, 525), 158, border=9)
    sticker(image, "06.png", (280, 0, 1000, 720), (1065, 472), 183, border=9)

    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((39, 34, 665, 125), radius=34, fill="#133961")
    draw.text((352, 78), "LEARN & PLAY!", font=face(58), anchor="mm", fill="white")
    colors = ["#F14B59", "#218FE4", "#9B5BE4", "#20A977", "#F38726", "#15ABC4"]
    for i, (letter, color) in enumerate(zip("ABCDEF", colors)):
        chunky_letter(image, letter, (151 + (i % 3) * 193, 252 + (i // 3) * 224), color)
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((93, 612, 613, 688), radius=34, fill="white")
    draw.text((353, 649), "SAY IT WITH ME!", font=face(47), anchor="mm", fill=NAVY)

    output = image.convert("RGB")
    output.save(HERE / "thumbnail.jpg", quality=95, subsampling=0, optimize=True)
    output.save(HERE / "thumbnail.png", optimize=True)
    output.resize((320, 180), Image.Resampling.LANCZOS).save(
        HERE / "thumbnail_mobile_preview.jpg", quality=95
    )
    assert (HERE / "thumbnail.jpg").stat().st_size < 2_000_000
    print(f"Thumbnail: {W}×{H}, {(HERE / 'thumbnail.jpg').stat().st_size // 1024} KB")


def export_metadata():
    metadata = json.loads((HERE / "metadata.json").read_text())
    tags = ", ".join(metadata["tags"])
    assert len(metadata["title"]) <= 100
    assert len(metadata["description"]) <= 5000
    assert len(tags) <= 500
    (HERE / "title.txt").write_text(metadata["title"] + "\n")
    (HERE / "description.txt").write_text(metadata["description"] + "\n")
    (HERE / "tags.txt").write_text(tags + "\n")
    text = (
        "# YouTube upload pack — Alphabet Adventure\n\n"
        "Use `alphabet_adventure_A-F_suno.mp4` with this pack.\n\n"
        "## Recommended title\n\n" + metadata["title"] + "\n\n"
        "## Alternative titles\n\n"
        + "\n".join(f"- {title}" for title in metadata["alternative_titles"])
        + "\n\n## Description — ready to paste\n\n" + metadata["description"]
        + "\n\n## Tags — ready to paste\n\n" + tags
        + "\n\n## Thumbnail\n\n"
        "Upload `thumbnail.jpg` (1280×720, RGB, under 2 MB). "
        "`thumbnail.png` is the lossless source; `thumbnail_mobile_preview.jpg` "
        "shows the design at 320×180 for readability review.\n\n"
        "## Upload settings\n\n"
        "- Audience: **Made for kids**\n"
        "- Language: **English**\n"
        "- Category: **Education**\n"
        "- Captions: use `../suno_captions.srt`\n\n"
        "## Notes\n\n"
        + "\n".join(f"- {note}" for note in metadata["notes"])
        + "\n\n## Regenerate\n\n"
        "From the repository root:\n\n"
        "```bash\n.venv/bin/python projects/alphabet_adventure/youtube/make_thumbnail.py\n```\n"
    )
    (HERE / "YOUTUBE.md").write_text(text)
    print(f"Metadata: title {len(metadata['title'])} characters, tags {len(tags)} characters")


if __name__ == "__main__":
    thumbnail()
    export_metadata()
