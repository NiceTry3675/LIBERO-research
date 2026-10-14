"""Figures for the 2-page paper (A4, two columns, 300 dpi).

Column width 85 mm = 1004 px, full width 175 mm = 2067 px.
Run with a Python that has PIL and numpy:
    python paper/figures/make_figures.py
All inputs are files already in the repo (run frames, replay renders, deck media).
No model calls.
"""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RUNS = ROOT / "outputs/robodawn_colab"
MEDIA = ROOT / "outputs/deck_media"
FONT = "/System/Library/Fonts/AppleSDGothicNeo.ttc"
FACE = {"regular": 0, "medium": 2, "semibold": 4, "bold": 6}
DPI = (300, 300)

INK = (11, 11, 11)
INK2 = (82, 81, 78)
WHITE = (255, 255, 255)


def font(px, weight="regular"):
    return ImageFont.truetype(FONT, px, index=FACE[weight])


def text_size(draw, s, f):
    x0, y0, x1, y1 = draw.textbbox((0, 0), s, font=f)
    return x1 - x0, y1 - y0, x0, y0


def fit(im, w, h):
    return im.resize((w, h), Image.LANCZOS)


def tag(draw, xy, s, f, fg=INK, bg=WHITE, pad=(7, 4), anchor="tl"):
    """Text in a filled box. xy is the box corner given by anchor (tl/tr/bl/br)."""
    tw, th, ox, oy = text_size(draw, s, f)
    bw, bh = tw + 2 * pad[0], th + 2 * pad[1]
    x, y = xy
    if "r" in anchor:
        x -= bw
    if "b" in anchor:
        y -= bh
    draw.rectangle([x, y, x + bw - 1, y + bh - 1], fill=bg)
    draw.text((x + pad[0] - ox, y + pad[1] - oy), s, font=f, fill=fg)
    return bw, bh


def letter(draw, xy, s):
    tag(draw, xy, s, font(34, "bold"), pad=(7, 5))


# ---------------------------------------------------------------- fig1
def fig1(named=True):
    W, H = 2067, 640
    m, gap = 4, 18
    canvas = Image.new("RGB", (W, H), WHITE)
    d = ImageDraw.Draw(canvas)

    # (a) what Gemini receives in one turn: perspective + top view with overlays
    ep = RUNS / "rt8b/robodawn/gemini_flash_flex/place_dual_shoes/shard_0/episode_000"
    aw, ah = 416, 312
    for k, cam in enumerate(["agent", "top"]):
        im = Image.open(ep / f"turn004_{cam}_camera.png").convert("RGB")
        canvas.paste(fit(im, aw, ah), (m, m + k * (ah + 8)))
    letter(d, (m, m), "(a)")

    # (c) knockover: place_shoe episode 5, turn-start agent images t02-t04
    ep5 = RUNS / "rt8b/robodawn/gemini_flash_flex/place_shoe/shard_5/episode_005"
    cw, ch = 273, 205
    box = (250, 130, 530, 340)  # 280 x 210 around shoe, gripper and target mat
    cx = W - m - cw
    ftag = font(30, "bold")
    for k, t in enumerate([2, 3, 4]):
        im = Image.open(ep5 / f"turn{t:03d}_agent_camera.png").convert("RGB").crop(box)
        y = m + k * (ch + 8)
        canvas.paste(fit(im, cw, ch), (cx, y))
        tag(d, (cx + cw, y + ch), f"t{t:02d}", ftag, fg=WHITE, bg=(0, 0, 0), pad=(6, 3), anchor="br")
    letter(d, (cx, m), "(c)")

    # (b) the 8 tasks, turn-1 raw renders (rt8d replay), order as in the deck sheet
    tasks = [("place_shoe", 0), ("place_bread_basket", 1), ("place_dual_shoes", 1),
             ("place_object_scale", 0), ("put_bottles_dustbin", 7), ("rotate_qrcode", 0),
             ("scan_object", 5), ("pick_diverse_bottles", 4)]
    bx0 = m + aw + gap
    bx1 = cx - gap
    tw = (bx1 - bx0 - 3 * 8) // 4
    lab = 36 if named else 0
    rgap = 14 if named else 8
    th = (H - 2 * m - rgap - 2 * lab) // 2
    # crop the 640x480 render to the tile aspect, trimming the sides equally
    cw_src = round(480 * tw / th)
    xoff = (640 - cw_src) // 2
    fname = font(30, "regular")
    for i, (task, e) in enumerate(tasks):
        r, c = divmod(i, 4)
        im = Image.open(RUNS / f"rt8d/replay/eval_baseline/{task}/episode_{e:03d}/images/c000_agent_camera_raw.jpg").convert("RGB")
        im = im.crop((xoff, 0, xoff + cw_src, 480))
        x = bx0 + c * (tw + 8)
        y = m + r * (th + lab + rgap)
        canvas.paste(fit(im, tw, th), (x, y))
        if named:
            s = task
            sw, sh, ox, oy = text_size(d, s, fname)
            assert sw <= tw, (s, sw, tw)
            d.text((x + (tw - sw) // 2 - ox, y + th + 5 - oy), s, font=fname, fill=INK)
    letter(d, (bx0, m), "(b)")

    name = "fig1_robodawn_robotwin.png" if named else "fig1_robodawn_robotwin_nonames.png"
    canvas.save(OUT / name, dpi=DPI)
    print(name, canvas.size, f"tile {tw}x{th}", f"src crop x {xoff}..{xoff + cw_src}")


# ---------------------------------------------------------------- fig2
# Post-skeptic counts over the 47 failures of the 8-task baseline
# (reports/실패_상한_재코딩.md:11-23; recounted from
# outputs/analysis/failure_ceiling/coding.json 'post_skeptic').
# (label, likely, likely+possible). Expected = 0.65*likely + 0.35*possible-only.
CEILING = [
    ("닫기 전 정렬", 1, 19),
    ("충돌 예측", 0, 16),
    ("놓기 위치", 4, 7),
    ("방향", 3, 5),
    ("숨은 조건", 0, 3),
    ("대상 정체", 1, 2),
    ("넘어짐 감지", 0, 1),
]
DETECT = 17  # about 17 episodes: smallest difference an 80-episode A/B can detect
C_LIKELY = (0x1C, 0x5C, 0xAB)
C_POSSIBLE = (0x86, 0xB6, 0xEF)
C_GRID = (0xE4, 0xE3, 0xDF)


def bar_end_rounded(d, x0, y0, x1, y1, fill, r=4):
    """Rectangle with only the right-hand (data-end) corners rounded."""
    if x1 - x0 <= 2 * r:
        d.rectangle([x0, y0, x1, y1], fill=fill)
        return
    d.rounded_rectangle([x0, y0, x1, y1], radius=r, fill=fill)
    d.rectangle([x0, y0, x0 + r, y1], fill=fill)


def diamond(d, cx, cy, r, fill, ring=WHITE, ring_w=2):
    d.polygon([(cx, cy - r - ring_w), (cx + r + ring_w, cy), (cx, cy + r + ring_w), (cx - r - ring_w, cy)], fill=ring)
    d.polygon([(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)], fill=fill)


def dashed_vline(d, x, y0, y1, fill, width=3, dash=12, gapl=8):
    y = y0
    while y < y1:
        d.line([(x, y), (x, min(y + dash, y1))], fill=fill, width=width)
        y += dash + gapl


def fig2():
    W = 1004
    f = font(30, "regular")
    fs = font(30, "regular")
    fb = font(30, "semibold")
    probe = ImageDraw.Draw(Image.new("RGB", (10, 10)))

    lab_w = max(text_size(probe, s, f)[0] for s, _, _ in CEILING)
    x0 = 10 + lab_w + 16          # plot origin
    scale = 38                     # px per episode
    xmax = 20
    head = 52                      # legend + threshold label row
    row_h, bar_h = 54, 30
    top = head + 12
    plot_h = row_h * len(CEILING)
    axis_y = top + plot_h + 4
    H = axis_y + 10 + 34 + 12 + 36 + 8
    im = Image.new("RGB", (W, H), WHITE)
    d = ImageDraw.Draw(im)
    X = lambda v: x0 + round(v * scale)

    # recessive grid + axis
    # no gridline at xmax: it would run through the '1/19' label at the end of the top bar
    for v in range(5, xmax, 5):
        d.line([(X(v), top), (X(v), axis_y)], fill=C_GRID, width=2)
    d.line([(x0, top - 4), (x0, axis_y)], fill=INK2, width=2)
    d.line([(x0, axis_y), (X(xmax), axis_y)], fill=INK2, width=2)
    for v in range(0, xmax + 1, 5):
        d.line([(X(v), axis_y), (X(v), axis_y + 8)], fill=INK2, width=2)
        s = str(v)
        sw, sh, ox, oy = text_size(d, s, fs)
        d.text((X(v) - sw // 2 - ox, axis_y + 12 - oy), s, font=fs, fill=INK2)
    s = "실패 에피소드 수 (47개 중, 정답 기준)"
    sw, sh, ox, oy = text_size(d, s, fs)
    d.text(((x0 + X(xmax)) // 2 - sw // 2 - ox, axis_y + 12 + 34 + 12 - oy), s, font=fs, fill=INK)

    # detection limit
    dashed_vline(d, X(DETECT), 8, axis_y, INK2)

    # bars
    for i, (name, lk, tot) in enumerate(CEILING):
        cy = top + i * row_h + row_h // 2
        y0, y1 = cy - bar_h // 2, cy + bar_h // 2
        sw, sh, ox, oy = text_size(d, name, f)
        d.text((x0 - 14 - sw - ox, cy - sh // 2 - oy), name, font=f, fill=INK)
        if lk:
            if tot > lk:
                d.rectangle([X(0) + 1, y0, X(lk) - 2, y1], fill=C_LIKELY)
            else:
                bar_end_rounded(d, X(0) + 1, y0, X(lk), y1, C_LIKELY)
        if tot > lk:
            bar_end_rounded(d, X(lk) + 1, y0, X(tot), y1, C_POSSIBLE)
        exp = 0.65 * lk + 0.35 * (tot - lk)
        diamond(d, X(exp), cy, 9, INK)
        s = f"{lk}/{tot}"
        sw, sh, ox, oy = text_size(d, s, fs)
        # white backing so a label that crosses the dashed limit line stays legible
        d.rectangle([X(tot) + 5, cy - sh // 2 - 6, X(tot) + 10 + sw + 5, cy + sh // 2 + 6], fill=WHITE)
        d.text((X(tot) + 10 - ox, cy - sh // 2 - oy), s, font=fs, fill=INK)

    # legend (left) and threshold label (right, ending at the line)
    lx = 10
    ly = 10
    for kind, label in [("likely", "유력"), ("possible", "가능"), ("exp", "기대 뒤집힘")]:
        if kind == "exp":
            diamond(d, lx + 12, ly + 15, 9, INK)
            lx += 30
        else:
            d.rectangle([lx, ly + 3, lx + 24, ly + 27], fill=C_LIKELY if kind == "likely" else C_POSSIBLE)
            lx += 32
        sw, sh, ox, oy = text_size(d, label, fs)
        d.text((lx - ox, ly + 15 - sh // 2 - oy), label, font=fs, fill=INK)
        lx += sw + 26
    s = f"80 에피소드 A/B 감지 한계 ≈ {DETECT}"
    sw, sh, ox, oy = text_size(d, s, fb)
    tx = X(DETECT) - 10 - sw
    assert tx > lx, (tx, lx)
    d.rectangle([tx - 4, ly - 2, X(DETECT) - 6, ly + 32], fill=WHITE)
    d.text((tx - ox, ly + 15 - sh // 2 - oy), s, font=fb, fill=INK2)

    im.save(OUT / "fig2_ceiling.png", dpi=DPI)
    print("fig2_ceiling.png", im.size, f"{im.size[1] / 300 * 25.4:.1f} mm tall")


# ---------------------------------------------------------------- fig3
# Exact Clef inputs. src/ holds copies of the monitor re-render (t29) and the
# claw retest crops (byte-identical to what Clef was sent, see deck result.json).
SRC = OUT / "src"
CLEF_EXAMPLES = [
    # (letter, title, [(image path, label), (image path, label)])
    ("(a)", "넘어짐", [
        (SRC / "monitor_sent_t29_START.png", "높이 25.0 cm"),
        (SRC / "monitor_sent_t29_NOW.png", "높이 8.1 cm"),
    ]),
    ("(b)", "정렬", [
        (SRC / "claw_sent_on_target.png", "0.0 cm → 그대로 0.41"),
        (SRC / "claw_sent_off_target_6p9cm.png", "6.9 cm → 그대로 0.50"),
    ]),
    ("(c)", "앞코 방향", [
        (RUNS / "rt9a/robodawn_dev/relay_baseline_gateclef_standard/place_dual_shoes/"
                "shard_26/episode_026/turn007_gate_left.jpg", "가려짐 → 오른쪽 0.68"),
        (ROOT / "outputs/analysis/clef_semantic/img/12039d170c3a_top.jpg", "보임 → 왼쪽 0.96"),
    ]),
]


def fig3(wide=False):
    fl = font(31, "bold")
    ft = font(30, "regular")
    fs = font(30, "regular")
    m = 8
    head, lab = 46, 40
    if wide:
        W = 2067
        ggap, tgap = 36, 10
        tile = (W - 2 * m - 2 * ggap - 3 * tgap) // 6
        H = m + head + tile + lab + 8
    else:
        W = 1004
        ggap, tgap = 22, 12
        tile = (W - 2 * m - 2 * ggap) // 3
        H = m + head + 2 * (tile + lab) + tgap + 8
    im = Image.new("RGB", (W, H), WHITE)
    d = ImageDraw.Draw(im)
    for g, (let, title, pair) in enumerate(CLEF_EXAMPLES):
        gx = m + g * ((2 * tile + tgap if wide else tile) + ggap)
        # group header: bold letter + title
        lw, lh, ox, oy = text_size(d, let, fl)
        d.text((gx - ox, m + 4 - oy), let, font=fl, fill=INK)
        tw_, th_, ox2, oy2 = text_size(d, title, ft)
        d.text((gx + lw + 10 - ox2, m + 4 + (lh - th_) // 2 - oy2 + 1), title, font=ft, fill=INK)
        for k, (path, label) in enumerate(pair):
            if wide:
                x, y = gx + k * (tile + tgap), m + head
            else:
                x, y = gx, m + head + k * (tile + lab + tgap)
            src = Image.open(path).convert("RGB")
            im.paste(fit(src, tile, tile), (x, y))
            sw, sh, ox, oy = text_size(d, label, fs)
            assert sw <= tile, (label, sw, tile)
            d.text((x + (tile - sw) // 2 - ox, y + tile + 6 - oy), label, font=fs, fill=INK)
    name = "fig3_clef_questions_wide.png" if wide else "fig3_clef_questions.png"
    im.save(OUT / name, dpi=DPI)
    print(name, im.size, f"tile {tile}", f"{im.size[1] / 300 * 25.4:.1f} mm tall")


def fig3_compact():
    """Column-width, one row: alignment (same item without and with the target ring) and toe examples.
    The marked crop is scripts/claw_marked_offline.marked_crop of eval_baseline/scan_object/0/left/c3; its answer
    (right 0.88, scored good) is in outputs/analysis/claw_marked/clef.jsonl."""
    groups = [("(a)", "정렬 (6.9 cm)", [
                  (SRC / "claw_sent_off_target_6p9cm.png", ["목표 표시 없음", "그대로 0.50 ×"]),
                  (SRC / "claw_marked_off_target_6p9cm.png", ["목표 표시", "오른쪽 0.88 ○"]),
              ]),
              # place_shoe episode 1, the same shoe: its release moment (clef_vantage P0, toe hidden, answer
              # right 0.62, truth left) and its arm-clear pre-grasp crop placed from the depth blob
              # (toe_armclear eval_baseline/place_shoe/1|c000|shoe, TOP, answer left 0.86, truth left)
              ("(b)", "앞코 방향 (같은 신발)", [
                  (SRC / "toe_release_hidden_shoe1.jpg", ["놓는 순간", "오른쪽 0.62 ×"]),
                  (SRC / "toe_armclear_pre_shoe1.png", ["잡기 전", "왼쪽 0.86 ○"]),
              ])]
    fl, ft, fs = font(31, "bold"), font(30, "regular"), font(29, "regular")
    W, m, ggap, tgap, head, line = 1004, 8, 22, 8, 46, 36
    tile = (W - 2 * m - ggap - 2 * tgap) // 4
    H = m + head + tile + 2 * line + 10
    im = Image.new("RGB", (W, H), WHITE)
    d = ImageDraw.Draw(im)
    for g, (let, title, pair) in enumerate(groups):
        gx = m + g * (2 * tile + tgap + ggap)
        lw, lh, ox, oy = text_size(d, let, fl)
        d.text((gx - ox, m + 4 - oy), let, font=fl, fill=INK)
        tw_, th_, ox2, oy2 = text_size(d, title, ft)
        d.text((gx + lw + 10 - ox2, m + 4 + (lh - th_) // 2 - oy2 + 1), title, font=ft, fill=INK)
        for k, (path, labels) in enumerate(pair):
            x, y = gx + k * (tile + tgap), m + head
            im.paste(fit(Image.open(path).convert("RGB"), tile, tile), (x, y))
            for j, s in enumerate(labels):
                sw, sh, ox, oy = text_size(d, s, fs)
                assert sw <= tile, (s, sw, tile)
                d.text((x + (tile - sw) // 2 - ox, y + tile + 6 + j * line - oy), s, font=fs, fill=INK)
    im.save(OUT / "fig3_compact.png", dpi=DPI)
    print("fig3_compact.png", im.size, f"tile {tile}", f"{im.size[1] / 300 * 25.4:.1f} mm tall")


if __name__ == "__main__":
    fig1(named=True)
    fig1(named=False)
    fig2()
    fig3(wide=False)
    fig3(wide=True)
    fig3_compact()
