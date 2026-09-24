"""Derives a browser palette (primary/secondary/atmosphere/fog) from the one
colour a skill declares, so adding a skill never means hand-tuning four hex
values in clients/browser.html. The formula is picked to reproduce the
palettes that were hand-tuned there before this existed (checked against
each of them; see skills/README.md), not from anywhere with special meaning.
"""

import colorsys


def _hex(r, g, b):
    return f"#{max(0, min(255, round(r))):02x}{max(0, min(255, round(g))):02x}{max(0, min(255, round(b))):02x}"


def derive_palette(rgb: tuple[int, int, int]) -> dict:
    r, g, b = (c / 255 for c in rgb)
    h, l, s = colorsys.rgb_to_hls(r, g, b)[0], colorsys.rgb_to_hls(r, g, b)[1], colorsys.rgb_to_hls(r, g, b)[2]

    def at(lightness, saturation=None):
        rr, gg, bb = colorsys.hls_to_rgb(h, lightness, s if saturation is None else saturation)
        return round(rr * 255), round(gg * 255), round(bb * 255)

    primary = rgb
    secondary = at(min(0.72, l + 0.24))
    atmosphere = at(min(0.82, l + 0.34))
    fog = at(0.95, min(0.55, s))
    return {
        "primaryHex": _hex(*primary), "primaryRGB": list(primary),
        "secondaryHex": _hex(*secondary), "secondaryRGB": list(secondary),
        "atmosphereHex": _hex(*atmosphere), "atmosphereRGB": list(atmosphere),
        "fogHex": _hex(*fog),
    }
