import cairosvg, io
from PIL import Image, ImageEnhance

# Sharp (right) glyph: brand blue->cyan gradient.
glyph_sharp = '''<svg width="128" height="128" viewBox="0 0 128 128" xmlns="http://www.w3.org/2000/svg">
  <defs><linearGradient id="grad" x1="0" y1="0" x2="1" y2="1">
    <stop offset="0" stop-color="#2f80ed"/><stop offset="0.55" stop-color="#33b5e5"/><stop offset="1" stop-color="#5af0e0"/>
  </linearGradient></defs>
  <rect x="14" y="24" width="100" height="80" rx="16" fill="none" stroke="url(#grad)" stroke-width="9"/>
  <path d="M52 44 L86 64 L52 84 Z" fill="url(#grad)"/>
</svg>'''
glyph_dull = '''<svg width="128" height="128" viewBox="0 0 128 128" xmlns="http://www.w3.org/2000/svg">
  <defs><linearGradient id="grad" x1="0" y1="0" x2="1" y2="1">
    <stop offset="0" stop-color="#8a6bd1"/><stop offset="1" stop-color="#6c4fb0"/>
  </linearGradient></defs>
  <rect x="14" y="24" width="100" height="80" rx="16" fill="none" stroke="url(#grad)" stroke-width="9"/>
  <path d="M52 44 L86 64 L52 84 Z" fill="url(#grad)"/>
</svg>'''
bg_svg = '''<svg width="128" height="128" viewBox="0 0 128 128" xmlns="http://www.w3.org/2000/svg">
  <defs><linearGradient id="bg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#0a2540"/><stop offset="1" stop-color="#0d1b2a"/></linearGradient>
  <clipPath id="r"><rect width="128" height="128" rx="26"/></clipPath></defs>
  <g clip-path="url(#r)"><rect width="128" height="128" fill="url(#bg)"/></g>
</svg>'''
# muted/neutral background for the OFF state
bg_off = '''<svg width="128" height="128" viewBox="0 0 128 128" xmlns="http://www.w3.org/2000/svg">
  <defs><linearGradient id="bg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#23272d"/><stop offset="1" stop-color="#16181c"/></linearGradient>
  <clipPath id="r"><rect width="128" height="128" rx="26"/></clipPath></defs>
  <g clip-path="url(#r)"><rect width="128" height="128" fill="url(#bg)"/></g>
</svg>'''

def render(svg, n):
    return Image.open(io.BytesIO(cairosvg.svg2png(bytestring=svg.encode(), output_width=n, output_height=n))).convert("RGBA")

def compose(size, blocks=16, active=True):
    ss = size * 8
    sharp = render(glyph_sharp, ss)
    dull = render(glyph_dull, ss)
    bg = render(bg_svg if active else bg_off, ss)
    pix = dull.resize((blocks, blocks), Image.NEAREST).resize((ss, ss), Image.NEAREST)
    half = ss // 2
    canvas = bg.copy()
    canvas.alpha_composite(ImageEnhance.Brightness(pix.crop((0, 0, half, ss))).enhance(0.92), (0, 0))
    canvas.alpha_composite(sharp.crop((half, 0, ss, ss)), (half, 0))
    out = canvas.resize((size, size), Image.LANCZOS)
    if not active:
        # desaturate to monochrome + slightly dim for the inactive state
        gray = ImageEnhance.Color(out).enhance(0.0)        # full desaturate
        gray = ImageEnhance.Brightness(gray).enhance(0.85) # dim a touch
        out = gray
    return out

if __name__ == "__main__":
    for s in (16, 32, 48, 128):
        compose(s, active=True).save(f"icons/icon-{s}.png")
        compose(s, active=False).save(f"icons/icon-off-{s}.png")
    compose(128, active=True).save("/tmp/on-128.png")
    compose(128, active=False).save("/tmp/off-128.png")
    print("rendered active + off icon sets")
