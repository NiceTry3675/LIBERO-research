# -*- coding: utf-8 -*-
"""Convert paper/draft.html to paper/draft.docx (A4, title block in one column, body in two columns).

Usage: python make_docx.py   (needs python-docx and Pillow)
"""
import re
from html.parser import HTMLParser
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Mm, Pt, RGBColor

HERE = Path(__file__).resolve().parent
SRC, OUT = HERE / "draft.html", HERE / "draft.docx"

KO_FONT, LATIN_FONT = "HY신명조", "Times New Roman"
PAGE_W, MARGIN_LR, COL_GAP = 210, 14, 6
TEXT_W = PAGE_W - 2 * MARGIN_LR
COL_W = (TEXT_W - COL_GAP) / 2
PT_PER_MM = 72 / 25.4
DOCX_TABLE_WIDTHS = [18, 28, 12, 42]  # percent of the column, table 1
PPR_AFTER_WORDWRAP = ("w:overflowPunct", "w:topLinePunct", "w:autoSpaceDE", "w:autoSpaceDN", "w:bidi",
                      "w:adjustRightInd", "w:snapToGrid", "w:spacing", "w:ind", "w:contextualSpacing",
                      "w:mirrorIndents", "w:suppressOverlap", "w:jc", "w:textDirection", "w:textAlignment",
                      "w:textboxTightWrap", "w:outlineLvl", "w:divId", "w:cnfStyle", "w:rPr", "w:sectPr",
                      "w:pPrChange")


# ---------- HTML -> blocks ----------

class Parser(HTMLParser):
    """Collects block elements with inline runs [(text, bold, italic)]."""

    BLOCK_TAGS = {"h1", "h2", "p", "figcaption"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.blocks, self.stack, self.cur = [], [], None
        self.bold = self.italic = 0
        self.in_refs = self.in_cols = False
        self.table = self.row = self.cell = None
        self.skip = 0

    def _classes(self, attrs):
        return (dict(attrs).get("class") or "").split()

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        cls = self._classes(attrs)
        if tag in ("style", "title", "head"):
            self.skip += 1
            return
        if tag == "div":
            self.stack.append(cls)
            if "cols" in cls:
                self.in_cols = True
                self.blocks.append({"kind": "cols_start"})
            if "refs" in cls:
                self.in_refs = True
            return
        if tag == "img":
            wide = any("wide" in c for c in self.stack[-1:]) or self._fig_wide
            self.blocks.append({"kind": "img", "src": a["src"], "wide": wide})
            return
        if tag == "figure":
            self._fig_wide = "wide" in cls
            return
        if tag == "table":
            self.table = []
            return
        if tag == "tr":
            self.row = []
            return
        if tag in ("td", "th"):
            self.cell = {"runs": [], "header": tag == "th", "rowspan": int(a.get("rowspan", 1))}
            return
        if tag == "br":
            self._add("\n")
            return
        if tag == "b":
            self.bold += 1
            return
        if tag == "i":
            self.italic += 1
            return
        if tag in self.BLOCK_TAGS:
            kind = tag
            if tag == "p" and cls:
                kind = cls[0]
            if tag == "h2" and self.in_refs:
                kind = "refs_h2"
            if tag == "p" and self.in_refs:
                kind = "ref"
            self.cur = {"kind": kind, "runs": []}

    def handle_endtag(self, tag):
        if tag in ("style", "title", "head"):
            self.skip -= 1
            return
        if tag == "div":
            cls = self.stack.pop() if self.stack else []
            if "refs" in cls:
                self.in_refs = False
            return
        if tag == "b":
            self.bold -= 1
            return
        if tag == "i":
            self.italic -= 1
            return
        if tag in ("td", "th"):
            self.row.append(self.cell)
            self.cell = None
            return
        if tag == "tr":
            self.table.append(self.row)
            self.row = None
            return
        if tag == "table":
            self.blocks.append({"kind": "table", "rows": self.table})
            self.table = None
            return
        if tag in self.BLOCK_TAGS and self.cur is not None:
            self.blocks.append(self.cur)
            self.cur = None

    _fig_wide = False

    def _add(self, text):
        target = self.cell["runs"] if self.cell is not None else (self.cur["runs"] if self.cur else None)
        if target is None:
            return
        target.append((text, self.bold > 0, self.italic > 0))

    def handle_data(self, data):
        if self.skip:
            return
        text = re.sub(r"\s+", " ", data)
        if text.strip() == "" and (self.cur is None and self.cell is None):
            return
        self._add(text)


def clean_runs(runs):
    out = [(t, b, i) for t, b, i in runs if t]
    if out:  # trim outer whitespace
        t, b, i = out[0]
        out[0] = (t.lstrip(), b, i)
        t, b, i = out[-1]
        out[-1] = (t.rstrip(), b, i)
    return [r for r in out if r[0]]


# ---------- docx helpers ----------

def set_run_font(run, size, bold=False, italic=False, color=None):
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.name = LATIN_FONT
    rpr = run._element.get_or_add_rPr()
    fonts = rpr.find(qn("w:rFonts"))
    if fonts is None:
        fonts = OxmlElement("w:rFonts")
        rpr.insert(0, fonts)
    for k in ("w:ascii", "w:hAnsi", "w:cs"):
        fonts.set(qn(k), LATIN_FONT)
    fonts.set(qn("w:eastAsia"), KO_FONT)
    if color:
        run.font.color.rgb = RGBColor.from_string(color)


def fmt(p, size, line, align=WD_ALIGN_PARAGRAPH.JUSTIFY, before=0, after=0, first=0, left=0, right=0,
        hanging=0, keep_next=False):
    pf = p.paragraph_format
    pf.alignment = align
    pf.line_spacing_rule = WD_LINE_SPACING.EXACTLY
    pf.line_spacing = Pt(line)
    pf.space_before, pf.space_after = Pt(before), Pt(after)
    pf.left_indent, pf.right_indent = Pt(left + hanging), Pt(right)
    pf.first_line_indent = Pt(-hanging if hanging else first)
    pf.keep_with_next = keep_next
    pf.widow_control = True
    # Korean text: break at word boundaries like the HTML (word-break: keep-all)
    ppr = p._element.get_or_add_pPr()
    wb = OxmlElement("w:wordWrap")
    wb.set(qn("w:val"), "0")
    ppr.insert_element_before(wb, *PPR_AFTER_WORDWRAP)
    # no automatic extra space between Hangul and Latin letters or digits (Word's East Asian default)
    for tag in ("w:autoSpaceDE", "w:autoSpaceDN"):
        el = OxmlElement(tag)
        el.set(qn("w:val"), "0")
        ppr.insert_element_before(el, *PPR_AFTER_WORDWRAP[PPR_AFTER_WORDWRAP.index(tag) + 1:])


def add_runs(p, runs, size, color=None, bold_all=False):
    for text, b, i in runs:
        parts = text.split("\n")
        for k, part in enumerate(parts):
            if k:
                p.add_run().add_break()
            if part:
                set_run_font(p.add_run(part), size, bold=b or bold_all, italic=i, color=color)


def set_cols(section, n, gap_mm):
    sect = section._sectPr
    cols = sect.find(qn("w:cols"))
    if cols is None:
        cols = OxmlElement("w:cols")
        sect.append(cols)
    cols.set(qn("w:num"), str(n))
    cols.set(qn("w:space"), str(int(gap_mm * 56.7)))


def cell_border(cell, **edges):
    tcpr = cell._element.get_or_add_tcPr()
    borders = tcpr.find(qn("w:tcBorders"))
    if borders is None:
        borders = OxmlElement("w:tcBorders")
        tcpr.append(borders)
    for edge, sz in edges.items():
        for old in borders.findall(qn(f"w:{edge}")):
            borders.remove(old)
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "single" if sz else "nil")
        if sz:
            el.set(qn("w:sz"), str(sz))
            el.set(qn("w:color"), "222222")
        borders.append(el)


def cell_margins(table, top=0.6, bottom=0.6, left=0.8, right=0.8):
    tblpr = table._element.tblPr
    mar = OxmlElement("w:tblCellMar")
    for edge, mm in (("top", top), ("left", left), ("bottom", bottom), ("right", right)):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:w"), str(int(mm * 56.7)))
        el.set(qn("w:type"), "dxa")
        mar.append(el)
    look = tblpr.find(qn("w:tblLook"))
    if look is not None:
        look.addprevious(mar)
    else:
        tblpr.append(mar)


# ---------- build ----------

def build():
    html = SRC.read_text(encoding="utf-8")
    # Word sets Korean wider than Chrome; these keep "물체 모니터" and the footnote marks on one line
    widths = DOCX_TABLE_WIDTHS
    parser = Parser()
    parser.feed(html)
    blocks = parser.blocks

    doc = Document()
    doc.core_properties.author = ""
    doc.core_properties.title = re.search(r"<title>(.*?)</title>", html).group(1)
    normal = doc.styles["Normal"]
    normal.font.size = Pt(9)
    normal.font.name = LATIN_FONT
    nrpr = normal.element.get_or_add_rPr()
    nfonts = nrpr.find(qn("w:rFonts"))
    if nfonts is None:
        nfonts = OxmlElement("w:rFonts")
        nrpr.insert(0, nfonts)
    for k in ("w:ascii", "w:hAnsi", "w:cs"):
        nfonts.set(qn(k), LATIN_FONT)
    nfonts.set(qn("w:eastAsia"), KO_FONT)
    lang = OxmlElement("w:lang")
    lang.set(qn("w:val"), "en-US")
    lang.set(qn("w:eastAsia"), "ko-KR")
    nrpr.append(lang)
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Mm(210), Mm(297)
    sec.top_margin, sec.bottom_margin = Mm(15), Mm(16)
    sec.left_margin = sec.right_margin = Mm(MARGIN_LR)
    sec.header_distance = sec.footer_distance = Mm(8)

    used_first = True  # the default template has no body paragraph

    def new_par():
        return doc.add_paragraph()

    prev_kind = None
    for blk in blocks:
        kind = blk["kind"]
        if kind == "cols_start":
            continue  # the column section starts after figure 1 (see "img" below)
        if kind == "h1":
            p = new_par()
            fmt(p, 14.5, 19.6, WD_ALIGN_PARAGRAPH.CENTER, after=2 * PT_PER_MM)
            add_runs(p, clean_runs(blk["runs"]), 14.5, bold_all=True)
        elif kind == "authors":
            p = new_par()
            fmt(p, 9.5, 13, WD_ALIGN_PARAGRAPH.CENTER)
            add_runs(p, clean_runs(blk["runs"]), 9.5)
        elif kind == "affil":
            p = new_par()
            fmt(p, 8.5, 12, WD_ALIGN_PARAGRAPH.CENTER, before=0.5 * PT_PER_MM, after=3.5 * PT_PER_MM)
            add_runs(p, clean_runs(blk["runs"]), 8.5, color="333333")
        elif kind == "abstract":
            p = new_par()
            fmt(p, 10, 14.2, left=6 * PT_PER_MM, right=6 * PT_PER_MM, after=3 * PT_PER_MM)
            add_runs(p, clean_runs(blk["runs"]), 10)
        elif kind == "img":
            path = HERE / blk["src"]
            p = new_par()
            fmt(p, 9, 12, WD_ALIGN_PARAGRAPH.CENTER, before=1.5 * PT_PER_MM, after=1.1 * PT_PER_MM, keep_next=True)
            pf = p.paragraph_format
            pf.line_spacing_rule = WD_LINE_SPACING.SINGLE
            p.add_run().add_picture(str(path), width=Mm(TEXT_W if blk["wide"] else COL_W))
            blk["_wide"] = blk["wide"]
        elif kind == "figcaption":
            p = new_par()
            fmt(p, 7.8, 10.9, after=(3 if prev_kind == "wide_img" else 2.2) * PT_PER_MM)
            add_runs(p, clean_runs(blk["runs"]), 7.8)
            if prev_kind == "wide_img":
                # end of the one-column title block: body continues in two columns
                body = doc.add_section(WD_SECTION.CONTINUOUS)
                set_cols(body, 2, COL_GAP)
                brk = doc.paragraphs[-1]
                fmt(brk, 1, 1, WD_ALIGN_PARAGRAPH.LEFT)
                used_first = True
        elif kind == "h2":
            p = new_par()
            fmt(p, 11, 15, WD_ALIGN_PARAGRAPH.LEFT, before=2.2 * PT_PER_MM, after=0.8 * PT_PER_MM, keep_next=True)
            add_runs(p, clean_runs(blk["runs"]), 11, bold_all=True)
        elif kind == "refs_h2":
            p = new_par()
            fmt(p, 11, 15, WD_ALIGN_PARAGRAPH.LEFT, before=2.2 * PT_PER_MM, after=0.8 * PT_PER_MM, keep_next=True)
            add_runs(p, clean_runs(blk["runs"]), 11, bold_all=True)
        elif kind == "p":
            p = new_par()
            fmt(p, 9, 12.9, first=0 if prev_kind in ("h2",) else 9)
            add_runs(p, clean_runs(blk["runs"]), 9)
        elif kind == "tcap":
            p = new_par()
            fmt(p, 7.8, 10.9, WD_ALIGN_PARAGRAPH.LEFT, before=1.2 * PT_PER_MM, after=0.6 * PT_PER_MM, keep_next=True)
            add_runs(p, clean_runs(blk["runs"]), 7.8)
        elif kind == "table":
            rows = blk["rows"]
            ncols = len(rows[0])
            t = doc.add_table(rows=len(rows), cols=ncols)
            t.alignment = WD_TABLE_ALIGNMENT.CENTER
            t.autofit = False
            cell_margins(t)
            col_mm = [COL_W * w / 100 for w in widths]
            for gc, w in zip(t._tbl.tblGrid.findall(qn("w:gridCol")), col_mm):
                gc.set(qn("w:w"), str(int(w * 56.7)))
            occupied = {}
            for r, hrow in enumerate(rows):
                c = 0
                for hcell in hrow:
                    while occupied.get((r, c)):
                        c += 1
                    cell = t.cell(r, c)
                    span = hcell["rowspan"]
                    if span > 1:
                        cell = cell.merge(t.cell(r + span - 1, c))
                        for k in range(1, span):
                            occupied[(r + k, c)] = True
                    p = cell.paragraphs[0]
                    fmt(p, 7.6, 10.3, WD_ALIGN_PARAGRAPH.LEFT, keep_next=True)
                    add_runs(p, clean_runs(hcell["runs"]), 7.6, bold_all=hcell["header"])
                    c += 1
            for r, row in enumerate(t.rows):
                trpr = row._tr.get_or_add_trPr()
                cant = OxmlElement("w:cantSplit")
                trpr.append(cant)
                for c, cell in enumerate(row.cells):
                    cell.width = Mm(col_mm[c])
                    top = 6 if r == 0 else 3
                    bottom = 6 if r == len(rows) - 1 else 0
                    cell_border(cell, top=top, bottom=bottom, left=0, right=0)
            # merged cells span to the last row: give them the bottom rule too
            for c in range(ncols):
                cell_border(t.cell(len(rows) - 1, c), bottom=6)
        elif kind == "tnote":
            p = new_par()
            fmt(p, 7.2, 9.7, before=0.5 * PT_PER_MM, after=2.2 * PT_PER_MM)
            add_runs(p, clean_runs(blk["runs"]), 7.2, color="333333")
        elif kind == "ref":
            p = new_par()
            fmt(p, 7.4, 10.2, WD_ALIGN_PARAGRAPH.LEFT, after=0.5 * PT_PER_MM, hanging=1.6 * 7.4)
            add_runs(p, clean_runs(blk["runs"]), 7.4)
        else:
            raise ValueError(f"unhandled block {kind}")
        prev_kind = "wide_img" if (kind == "img" and blk["wide"]) else kind

    doc.save(OUT)
    print(OUT, len(blocks), "blocks")


if __name__ == "__main__":
    build()
