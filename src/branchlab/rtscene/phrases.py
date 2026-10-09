"""The single vocabulary that turns parser measurements into text for a model.

Rules:
1. Every comparison is made here, in code, and stated as its result word; the number follows in parentheses
   (the 'numbers' format keeps only the numbers).
2. Directions are words relative to the robot (left/right as the robot faces the table, near/far from it),
   never axis signs to compare.
3. Facts describe, they never state a verdict.
4. A fact whose inputs are missing is left out, not guessed.
"""
from __future__ import annotations

from collections import Counter

from . import colour

SIZE = [(4, "tiny"), (10, "small"), (18, "medium"), (30, "large"), (1e9, "very large")]
HEIGHT = [(1.5, "flat"), (5, "low"), (12, "medium height"), (1e9, "tall")]


def _bin(value: float, table) -> str:
    for limit, word in table:
        if value < limit:
            return word
    return table[-1][1]


def shape_word(b) -> str:
    if b.flat or b.h95 < 1.5:
        return "flat round sheet" if b.circularity > 0.8 else "flat sheet"
    if b.hollow >= 0.3 and b.circularity > 0.7:
        return "open round container"
    if b.length / max(b.width, 0.1) > 2.5:
        return "long narrow object"
    if b.fill > 0.85 and (b.top_tilt is None or b.top_tilt < 10):
        return "box-shaped object"
    if b.circularity > 0.8:
        return "round object"
    return "irregular object"


def side_word(x: float) -> str:
    return "left" if x < -15 else "centre-left" if x < -5 else "centre" if x <= 5 else "centre-right" if x <= 15 else "right"


def depth_word(y: float) -> str:
    return "near the robot" if y < -15 else "in the middle of the table" if y < 5 else "far from the robot"


def describe(blobs, white_l: float) -> dict[int, dict]:
    """blob id -> the words and numbers every format draws from."""
    out = {}
    for b in blobs:
        out[b.id] = {"colour": colour.name(b.lab, white_l), "size": _bin(max(b.length, b.width), SIZE),
                     "height": _bin(b.h95, HEIGHT), "shape": shape_word(b), "side": side_word(b.centroid[0]),
                     "depth": depth_word(b.centroid[1]), "length": b.length, "width": b.width, "h95": b.h95,
                     "x": b.centroid[0], "y": b.centroid[1], "lab": b.lab}
    colours = Counter(d["colour"] for d in out.values())
    if out:
        tallest = max(out, key=lambda k: out[k]["h95"])
        largest = max(out, key=lambda k: out[k]["length"] * out[k]["width"])
        flats = [k for k in out if out[k]["height"] == "flat"]
        for k, d in out.items():
            rel = []
            if colours[d["colour"]] == 1:
                rel.append(f"the only {d['colour']} object")
            if k == tallest:
                rel.append("the tallest object")
            if k == largest:
                rel.append("the largest footprint")
            if len(flats) > 1 and k in flats and k == max(flats, key=lambda f: out[f]["length"] * out[f]["width"]):
                rel.append("the largest flat object")
            d["relations"] = rel
    return out


def line_words(n: int, d: dict, relations: bool = False) -> str:
    text = (f"O{n}: {d['colour']}, {d['size']} ({d['length']:.0f} x {d['width']:.0f} cm footprint), {d['height']} "
            f"({d['h95']:.0f} cm high), {d['shape']}; {d['side']}, {d['depth']}")
    if relations and d.get("relations"):
        text += "; " + ", ".join(d["relations"])
    return text


def line_numbers(n: int, d: dict) -> str:
    L, a, b = d["lab"]
    return (f"O{n}: colour Lab ({L:.0f}, {a:.0f}, {b:.0f}); footprint {d['length']:.1f} x {d['width']:.1f} cm; "
            f"height {d['h95']:.1f} cm; position x = {d['x']:.0f} cm, y = {d['y']:.0f} cm")


FRAME_NUMBERS = ("Positions: x grows to the robot's right, y grows away from the robot; the table centre is (0, 0). "
                 "Colours are CIE Lab (L lightness 0-100; a: green - / red +; b: blue - / yellow +).")


def scene_text(numbered: list[tuple[int, dict]], fmt: str) -> str:
    """fmt: words | relations | numbers | minimal (object numbers only, for image-only questions)."""
    if fmt == "minimal":
        return "Objects on the table are marked O1..O%d in the image." % len(numbered)
    if fmt == "numbers":
        return "\n".join([FRAME_NUMBERS] + [line_numbers(n, d) for n, d in numbered])
    return "\n".join(line_words(n, d, relations=(fmt == "relations")) for n, d in numbered)
