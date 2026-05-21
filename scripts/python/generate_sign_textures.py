import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def find_font(size):
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for c in candidates:
        if Path(c).exists():
            return ImageFont.truetype(c, size)
    return ImageFont.load_default()


def draw_sign(value: str, size: int = 512) -> Image.Image:
    super_size = size * 4
    img = Image.new("RGBA", (super_size, super_size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    center = super_size // 2
    outer_r = int(super_size * 0.47)
    ring = int(super_size * 0.12)
    draw.ellipse([center - outer_r, center - outer_r, center + outer_r, center + outer_r],
                 fill=(208, 22, 30, 255))
    draw.ellipse([center - outer_r + ring, center - outer_r + ring,
                  center + outer_r - ring, center + outer_r - ring],
                 fill=(252, 252, 250, 255))

    if len(value) == 2:
        font_size = int(super_size * 0.48)
    else:
        font_size = int(super_size * 0.39)
    font = find_font(font_size)
    bbox = draw.textbbox((0, 0), value, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    tx = center - tw / 2 - bbox[0]
    ty = center - th / 2 - bbox[1] - int(super_size * 0.025)
    draw.text((tx, ty), value, fill=(8, 8, 14, 255), font=font)
    return img.resize((size, size), Image.LANCZOS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="src/main/resources/signs")
    ap.add_argument("--size", type=int, default=512)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for v in ("50", "70", "90", "110"):
        img = draw_sign(v, args.size)
        path = out_dir / f"{v}.png"
        img.save(path, "PNG")
        print(f"wrote {path}  ({args.size}x{args.size})")


if __name__ == "__main__":
    main()
