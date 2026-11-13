"""YouTube thumbnail (1280x720) from scene 11: big Hindi title on top, English tag line.

Run from the repo root:  .venv/bin/python projects/brush_brush_karo/youtube/make_thumbnail.py
"""
import os
import sys

from PIL import Image, ImageDraw, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT)

from img2vid.render import _font  # noqa: E402  picks a bold Devanagari font

W, H = 1280, 720
TITLE = "ब्रश ब्रश करो!"
TAG = "Brushing Song for Kids"


def outlined(draw, xy, text, font, fill, stroke, width, anchor="mm"):
    draw.text(xy, text, font=font, fill=fill, stroke_width=width, stroke_fill=stroke,
              anchor=anchor)


def main():
    src = Image.open(os.path.join(HERE, "..", "images", "11.png")).convert("RGB")
    # Crop slightly lower so the title band sits over the wall, not the faces.
    sw, sh = src.size
    crop_h = int(sw * H / W)
    img = src.crop((0, sh - crop_h, sw, sh)).resize((W, H), Image.LANCZOS)

    # Soft dark band at the top so the title pops on any background.
    band = Image.new("L", (W, H), 0)
    ImageDraw.Draw(band).rectangle((0, 0, W, 190), fill=150)
    band = band.filter(ImageFilter.GaussianBlur(40))
    img = Image.composite(Image.new("RGB", (W, H), (40, 10, 60)), img, band)

    d = ImageDraw.Draw(img)
    title_font = _font(TITLE, 128, None)
    outlined(d, (W // 2, 92), TITLE, title_font, (255, 226, 40), (60, 20, 90), 10)

    tag_font = _font(TAG, 46, None)
    tw = d.textlength(TAG, font=tag_font)
    x0, y0 = (W - tw) // 2 - 28, H - 92
    d.rounded_rectangle((x0, y0, x0 + tw + 56, y0 + 66), radius=33, fill=(255, 64, 129))
    outlined(d, (W // 2, y0 + 33), TAG, tag_font, (255, 255, 255), (150, 20, 70), 3)

    out = os.path.join(HERE, "thumbnail.jpg")
    img.save(out, quality=92)
    print(f"wrote {out} ({os.path.getsize(out) // 1024} KB)")


if __name__ == "__main__":
    main()
