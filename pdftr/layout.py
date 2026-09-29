"""Translated PDF with the original layout: source text is removed and the translation is drawn into
the same boxes. Pictures, vector drawings, table rulings and backgrounds are left untouched."""

from __future__ import annotations

import html
import math
import re
import statistics
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Callable

import pymupdf

from . import markdown, ocr
from .markdown import Block, has_letters

# Paragraph starts that must not be glued to the previous line: bullets, "1.", "a)", "(2)"
_LIST_START = re.compile(r"^([-–•▪►●◦■□✓]|\(?\d{1,3}[.)]|\(?[a-zA-Z][.)])\s")
_SENT_END = re.compile(r"[.!?:;。！？：；]$")
_MD_BOLD = re.compile(r"\*\*(.+?)\*\*|__(.+?)__")
_TAG = re.compile(r"<[^>]+>")
_SAMPLE_DPI = 100


@dataclass
class Region:
    """One box of text on a page, translated and drawn as a unit."""

    rect: pymupdf.Rect
    blocks: list[Block]
    size: float = 0.0  # font size in pt; 0 = guess from box area and text length
    color: tuple[float, float, float] | None = None  # None = pick from the page image
    bold: bool = False
    leading: float = 1.15  # line height as a multiple of the font size
    paint: bool = False  # cover the original with its background colour (scans)
    erase: list[pymupdf.Rect] = field(default_factory=list)  # text-layer glyphs to remove
    from_markdown: bool = False  # OCR text: may carry **bold**, <br> etc.

    def texts(self) -> list[str]:
        return markdown.translatable_texts(self.blocks)


# ---------------------------------------------------------------- text layer

@dataclass
class _Line:
    rect: pymupdf.Rect
    text: str
    size: float
    color: int
    bold: bool


def _is_bold(span: dict) -> bool:
    return bool(span["flags"] & 16) or "bold" in span["font"].lower()


def _lines(page: pymupdf.Page) -> list[_Line]:
    """Horizontal text lines, split where spans are far apart (table cells printed as one line)."""
    out = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            dx, dy = line["dir"]
            if abs(dy) > 0.05 or dx < 0:  # rotated or vertical text stays as is
                continue
            segs: list[list[dict]] = []
            for span in line["spans"]:
                if not span["text"].strip():
                    continue
                if segs and span["bbox"][0] - segs[-1][-1]["bbox"][2] > 1.5 * span["size"]:
                    segs.append([span])
                elif segs:
                    segs[-1].append(span)
                else:
                    segs.append([span])
            for spans in segs:
                weights = [len(s["text"].strip()) or 1 for s in spans]
                main = spans[weights.index(max(weights))]
                rect = pymupdf.Rect(spans[0]["bbox"])
                for s in spans[1:]:
                    rect |= s["bbox"]
                out.append(_Line(
                    rect=rect,
                    text="".join(s["text"] for s in spans).strip(),
                    size=sum(s["size"] * w for s, w in zip(spans, weights)) / sum(weights),
                    color=main["color"],
                    bold=sum(w for s, w in zip(spans, weights) if _is_bold(s)) * 2 > sum(weights),
                ))
    return out


def _graphics(page: pymupdf.Page) -> tuple[list[tuple[float, float, float]], list[pymupdf.Rect]]:
    """Horizontal rulings (y, x0, x1) that table rows must not be merged across,
    and obstacles (lines, box edges, curves, pictures) that translated text must not grow into."""
    rulings, obstacles = [], [pymupdf.Rect(i["bbox"]) for i in page.get_image_info()]
    try:
        drawings = page.get_drawings()
    except Exception:
        return rulings, obstacles
    for d in drawings:
        for item in d["items"]:
            if item[0] == "l":
                p, q = item[1], item[2]
                if abs(p.y - q.y) < 1 and abs(p.x - q.x) > 3:
                    rulings.append((p.y, min(p.x, q.x), max(p.x, q.x)))
                obstacles.append(pymupdf.Rect(min(p.x, q.x), min(p.y, q.y), max(p.x, q.x) + 0.5, max(p.y, q.y) + 0.5))
            elif item[0] in ("re", "qu"):
                r = item[1] if item[0] == "re" else item[1].rect
                if r.width > 3:
                    rulings += [(r.y0, r.x0, r.x1), (r.y1, r.x0, r.x1)]
                obstacles += [pymupdf.Rect(r.x0, r.y0, r.x1, r.y0 + 0.5), pymupdf.Rect(r.x0, r.y1 - 0.5, r.x1, r.y1),
                              pymupdf.Rect(r.x0, r.y0, r.x0 + 0.5, r.y1), pymupdf.Rect(r.x1 - 0.5, r.y0, r.x1, r.y1)]
            else:  # curves: circles, arrows, outlines of parts
                obstacles.append(pymupdf.Rect(d["rect"]))
                break
    return rulings, obstacles


def _grow(rect: pymupdf.Rect, single_line: bool, obstacles: list[pymupdf.Rect],
          content: pymupdf.Rect) -> pymupdf.Rect:
    """Room for a longer translation: a single line grows to the right, a paragraph grows down,
    up to the nearest obstacle and not past the text area of the page."""
    r = pymupdf.Rect(rect)
    others = [o for o in obstacles if not o.intersects(rect)]
    if single_line:
        pad = 0.2 * rect.height
        x1 = content.x1
        for o in others:
            if o.y0 < rect.y1 - pad and o.y1 > rect.y0 + pad and o.x0 >= rect.x1 - 1:
                x1 = min(x1, o.x0 - 2)
        r.x1 = max(rect.x1, x1)
    else:
        y1 = content.y1
        for o in others:
            if o.x0 < rect.x1 - 1 and o.x1 > rect.x0 + 1 and o.y0 >= rect.y1 - 1:
                y1 = min(y1, o.y0 - 2)
        r.y1 = max(rect.y1, y1)
    return r


def _continues(group: list[_Line], ln: _Line, rulings) -> bool:
    last = group[-1]
    s = last.size
    if abs(ln.size - s) > 0.15 * s or ln.bold != last.bold or ln.color != last.color:
        return False
    if ln.rect.y0 < last.rect.y0 + 0.5 * last.rect.height:  # same row: a neighbouring cell or column
        return False
    if ln.rect.y0 - last.rect.y1 > 0.6 * s:  # paragraph gap
        return False
    if min(ln.rect.x1, last.rect.x1) - max(ln.rect.x0, last.rect.x0) <= 0:
        return False
    if _LIST_START.match(ln.text):
        return False
    right = max(l.rect.x1 for l in group)
    if last.rect.x1 < right - 2 * s and _SENT_END.search(last.text):  # short last line ends a paragraph
        return False
    top, bottom = (last.rect.y0 + last.rect.y1) / 2, (ln.rect.y0 + ln.rect.y1) / 2
    x0, x1 = min(last.rect.x0, ln.rect.x0), max(last.rect.x1, ln.rect.x1)
    return not any(top < y < bottom and rx0 < x1 and rx1 > x0 for y, rx0, rx1 in rulings)


def _join(texts: list[str]) -> str:
    out = ""
    for t in texts:
        if out.endswith("-") and len(out) > 1 and out[-2].isalpha() and t[:1].islower():
            out = out[:-1] + t  # word hyphenated at the line end
        else:
            out = markdown.join_lines([out, t])
    return out


def _invisible_text(page: pymupdf.Page) -> bool:
    """An OCR'd scan: the visible text is in the image, the text layer is invisible (render mode 3)."""
    try:
        trace = page.get_texttrace()
    except Exception:
        return False
    total = sum(len(s["chars"]) for s in trace)
    hidden = sum(len(s["chars"]) for s in trace if s.get("type") == 3)
    return total > 0 and hidden * 2 > total


def text_regions(page: pymupdf.Page) -> list[Region]:
    lines = _lines(page)
    rulings, obstacles = _graphics(page)
    obstacles += [l.rect for l in lines]
    content = pymupdf.Rect()
    for l in lines:
        content |= l.rect
    # right margin as wide as the left one: the widest line may still be shorter than the text area
    content.x1 = max(content.x1, page.rect.x1 - (content.x0 - page.rect.x0))
    groups: list[list[_Line]] = []
    for ln in lines:
        # a few recent groups are candidates: columns and table cells interleave in extraction order
        target = next((g for g in reversed(groups[-12:]) if _continues(g, ln, rulings)), None)
        if target:
            target.append(ln)
        else:
            groups.append([ln])

    scan = _invisible_text(page)
    out = []
    for g in groups:
        text = _join([l.text for l in g])
        if sum(1 for c in text if c.isalpha()) < 2:  # page numbers, callout digits, single letters
            continue
        rect = pymupdf.Rect(g[0].rect)
        for l in g[1:]:
            rect |= l.rect
        erase = []
        for l in g:
            pad = 0.2 * l.rect.height  # keep the glyph centres of neighbouring lines out
            erase.append(pymupdf.Rect(l.rect.x0, l.rect.y0 + pad, l.rect.x1, l.rect.y1 - pad))
        c = g[0].color
        size = statistics.median(l.size for l in g)
        leading = 1.15
        if len(g) > 1:
            leading = min(max((g[-1].rect.y0 - g[0].rect.y0) / (len(g) - 1) / size, 1.0), 2.0)
        out.append(Region(
            # on scans the "free" space is part of the image, painting over it would hide the picture
            rect=rect if scan else _grow(rect, len(g) == 1, obstacles, content),
            blocks=[Block("para", text=text)],
            size=size,
            leading=leading,
            color=None if scan else ((c >> 16 & 255) / 255, (c >> 8 & 255) / 255, (c & 255) / 255),
            bold=g[0].bold,
            paint=scan,
            erase=erase,
        ))
    return out


# ---------------------------------------------------------------- OCR pages

def _layer_metrics(lines: list[_Line], rect: pymupdf.Rect) -> tuple[float, float]:
    """(font size, line height factor) from text-layer lines inside an OCR box; (0, 1.15) if none."""
    inside = [l for l in lines if rect.contains((l.rect.tl + l.rect.br) / 2)]
    if not inside:
        return 0.0, 1.15
    size = statistics.median(l.size for l in inside)
    tops = sorted({round(l.rect.y0) for l in inside})
    steps = [b - a for a, b in zip(tops, tops[1:]) if 0.9 * size < b - a < 2.5 * size]
    return size, min(max(statistics.median(steps) / size, 1.0), 2.0) if steps else 1.15


def ocr_regions(page: pymupdf.Page, raw: str) -> list[Region]:
    r0 = page.rect
    # Pictures smaller than the page: a text box lying inside one is part of the picture (a logo, a label)
    pictures = [pymupdf.Rect(i["bbox"]) for i in page.get_image_info()]
    pictures = [p for p in pictures if p.width * p.height < 0.5 * r0.width * r0.height]
    lines = _lines(page)  # a hybrid PDF may keep part of the text in a text layer: it knows the real size
    out = []
    for reg in ocr.parse_regions(raw):
        if reg.label in ocr.IMAGE_LABELS or reg.label in ocr.FORMULA_LABELS:
            continue
        blocks = markdown.parse(reg.content)
        if not markdown.translatable_texts(blocks):
            continue
        x0, y0, x1, y1 = reg.box
        rect = pymupdf.Rect(r0.x0 + x0 * r0.width, r0.y0 + y0 * r0.height,
                            r0.x0 + x1 * r0.width, r0.y0 + y1 * r0.height)
        # a picture much larger than the text is a logo or a drawing; one that fits the text tightly
        # is a word drawn as a picture (see ingest._text_as_images) and gets translated
        if any(p.contains(rect + (2, 2, -2, -2)) and p.width * p.height > 2 * rect.width * rect.height
               for p in pictures):
            continue
        size, leading = _layer_metrics(lines, rect)
        out.append(Region(rect=rect, blocks=blocks, size=size, leading=leading,
                          bold=reg.label in ocr.TITLE_LABELS, paint=True, erase=[rect], from_markdown=True))
    return out


def extract(pdf: Path, pages: list[int], ocr_raw: dict[int, str]) -> tuple[pymupdf.Document, list[list[Region]]]:
    """Open a copy of the PDF with only `pages` and find the text regions on each of them."""
    doc = pymupdf.open(pdf)
    if pages != list(range(doc.page_count)):
        doc.select(pages)
    regions = []
    for n, index in enumerate(pages):
        page = doc[n]
        if page.rotation:
            page.remove_rotation()  # draw in what-you-see coordinates, OCR boxes are in those too
        if index in ocr_raw:
            regs = ocr_regions(page, ocr_raw[index])
            if not regs and has_letters(ocr_raw[index]):
                print(f"  ! стр. {index + 1}: OCR не дал координат блоков, в PDF она останется без перевода "
                      f"(координаты даёт профиль deepseek-ocr)", flush=True)
        else:
            regs = text_regions(page)
        regions.append(regs)
    return doc, regions


# ---------------------------------------------------------------- drawing

@dataclass
class Fonts:
    family: str
    face_css: str = ""
    archive: pymupdf.Archive | None = None

    @classmethod
    def load(cls, regular: str | None, bold: str | None) -> "Fonts":
        """TTF files for the translation; without them MuPDF's built-in fonts are used."""
        if not regular or not Path(regular).is_file():
            return cls("sans-serif")
        archive = pymupdf.Archive()
        css = []
        for path, weight in ((regular, "normal"), (bold, "bold")):
            if path and Path(path).is_file():
                archive.add(str(Path(path).parent))
                css.append(f"@font-face {{font-family: tr; font-weight: {weight}; src: url({Path(path).name});}}")
        return cls("tr", "\n".join(css), archive)


def _hex(rgb) -> str:
    return "#" + "".join(f"{max(0, min(255, round(c * 255))):02x}" for c in rgb)


def _sample(pix: pymupdf.Pixmap, rect: pymupdf.Rect) -> tuple[tuple, tuple]:
    """(background, text colour) of a box on the rendered page, as 0..1 RGB."""
    k = _SAMPLE_DPI / 72
    ir = pymupdf.IRect(rect.x0 * k, rect.y0 * k, rect.x1 * k, rect.y1 * k) & pix.irect
    outer = pymupdf.IRect(ir.x0 - 2, ir.y0 - 2, ir.x1 + 2, ir.y1 + 2) & pix.irect
    if ir.is_empty or outer.is_empty:
        return (1, 1, 1), (0, 0, 0)

    def px(x, y):
        return pix.pixel(min(x, pix.width - 1), min(y, pix.height - 1))[:3]

    step = max(1, (outer.width + outer.height) // 200)
    ring = [px(x, y) for x in range(outer.x0, outer.x1, step) for y in (outer.y0, outer.y1 - 1)]
    ring += [px(x, y) for y in range(outer.y0, outer.y1, step) for x in (outer.x0, outer.x1 - 1)]
    bg = tuple(statistics.median(p[i] for p in ring) for i in range(3))

    step = max(1, int(math.sqrt(ir.width * ir.height / 4000)))
    inner = [px(x, y) for x in range(ir.x0, ir.x1, step) for y in range(ir.y0, ir.y1, step)]
    dist = [sum(abs(a - b) for a, b in zip(p, bg)) for p in inner]
    top = max(dist, default=0)
    if top < 90:
        fg = (0, 0, 0) if sum(bg) > 3 * 128 else (255, 255, 255)
    else:
        ink = [p for p, d in zip(inner, dist) if d >= 0.6 * top]
        fg = tuple(sum(p[i] for p in ink) / len(ink) for i in range(3))
    return tuple(c / 255 for c in bg), tuple(c / 255 for c in fg)


def _guess_size(region: Region) -> float:
    """Font size at which the source text would roughly fill the box."""
    text = " ".join(region.texts())
    wide = sum(1 for c in text if markdown._CJK_CHAR.match(c))
    units = 0.6 * (len(text) - wide) + 1.2 * wide
    size = math.sqrt(0.85 * region.rect.width * region.rect.height / max(units, 1))
    rows = sum(len(b.rows) if b.kind == "table" else len(markdown.html_table_rows(b.raw))
               for b in region.blocks if b.kind in ("table", "html_table"))
    if rows:  # cell padding and borders take a good part of each row
        size = min(size, region.rect.height / rows / 1.6)
    return max(5.0, min(size, region.rect.height / 1.15, 36.0 if region.bold else 24.0))


def _ink_metrics(pix: pymupdf.Pixmap, rect: pymupdf.Rect, bg) -> tuple[float, float]:
    """(font size, line height factor) of a scanned text box, from the rows of pixels that carry ink.
    Table rulings are ignored: vertical ones add the same count to every row, horizontal ones are
    one or two pixels tall. (0, 1.15) if no text lines are found."""
    k = _SAMPLE_DPI / 72
    ir = pymupdf.IRect(rect.x0 * k, rect.y0 * k, rect.x1 * k, rect.y1 * k) & pix.irect
    if ir.is_empty:
        return 0.0, 1.15
    # narrow vertical strips: in a table, lines of different columns sit at different heights
    # and would otherwise fill every row of the box
    n = max(1, min(8, ir.width // 60))
    runs, steps = [], []
    for i in range(n):
        xs = range(ir.x0 + i * ir.width // n, ir.x0 + (i + 1) * ir.width // n, 2)
        counts = [sum(1 for x in xs if sum(abs(a - b * 255) for a, b in zip(pix.pixel(x, y)[:3], bg)) > 120)
                  for y in range(ir.y0, ir.y1)]
        base = min(counts)
        strip, start = [], None
        for y, c in enumerate(counts + [base]):
            if c > base + 1 and start is None:
                start = y
            elif c <= base + 1 and start is not None:
                strip.append((start, y))
                start = None
        strip = [(a, b) for a, b in strip if b - a >= 4]
        runs += strip
        steps += [b[0] - a[0] for a, b in zip(strip, strip[1:])]
    if not runs:
        return 0.0, 1.15
    # ink of a line spans roughly cap height + descenders, ~0.9 of the font size for CJK, ~0.85 for Latin
    size = statistics.median(b - a for a, b in runs) / 0.88 / k
    steps = [s / k for s in steps if s / k < 2.5 * size]
    leading = min(max(statistics.median(steps) / size, 1.0), 2.0) if steps else 1.15
    return max(4.0, min(size, 40.0)), leading


def _blank(pix: pymupdf.Pixmap, bg, x0: int, y0: int, x1: int, y1: int) -> bool:
    """Whether the pixel box (in pixmap coordinates) is plain background."""
    for x in range(max(x0, 0), min(x1, pix.width), 2):
        for y in range(max(y0, 0), min(y1, pix.height), 2):
            if sum(abs(a - b * 255) for a, b in zip(pix.pixel(x, y)[:3], bg)) > 60:
                return False
    return True


def _grow_blank(r: Region, pix: pymupdf.Pixmap, bg, others: list[pymupdf.Rect], size: float) -> pymupdf.Rect:
    """On a scan, room for a longer translation is where the image is empty background:
    a one-line box grows to the right, a taller one grows down, never into other regions."""
    k = _SAMPLE_DPI / 72
    rect = pymupdf.Rect(r.rect)
    x0, y0, x1, y1 = (round(v * k) for v in rect)
    margin_x, margin_y = int(0.05 * pix.width), int(0.04 * pix.height)
    if rect.height < 2 * size * r.leading:
        pad = max(1, (y1 - y0) // 5)
        x = x1
        while x + 4 <= pix.width - margin_x and _blank(pix, bg, x, y0 + pad, x + 4, y1 - pad):
            x += 4
        grown = pymupdf.Rect(rect.x0, rect.y0, max(rect.x1, (x - 3) / k), rect.y1)
    else:
        y = y1
        while y + 4 <= pix.height - margin_y and _blank(pix, bg, x0, y, x1, y + 4):
            y += 4
        grown = pymupdf.Rect(rect.x0, rect.y0, rect.x1, max(rect.y1, (y - 3) / k))
    for o in others:
        if grown.intersects(o) and not rect.intersects(o):
            if grown.x1 > rect.x1:
                grown.x1 = max(rect.x1, min(grown.x1, o.x0 - 1))
            if grown.y1 > rect.y1:
                grown.y1 = max(rect.y1, min(grown.y1, o.y0 - 1))
    return grown


def _inline(text: str, from_markdown: bool) -> str:
    if from_markdown:
        text = _TAG.sub(" ", text)
    text = html.escape(text, quote=False)
    if from_markdown:
        text = _MD_BOLD.sub(lambda m: f"<b>{m.group(1) or m.group(2)}</b>", text)
    return text


def _table(rows: list[list[str]], md: bool) -> str:
    cells = "".join("<tr>" + "".join(f"<td>{_inline(c, md)}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table>{cells}</table>"


class _TableCleaner(HTMLParser):
    """OCR table HTML reduced to table/tr/td/th with colspan/rowspan; cell text escaped."""

    def __init__(self, md: bool):
        super().__init__()
        self.md = md
        self.out: list[str] = []
        self.cell: list[str] | None = None

    def handle_starttag(self, tag, attrs):
        if tag in ("td", "th"):
            spans = "".join(f' {k}="{int(v)}"' for k, v in attrs
                            if k in ("colspan", "rowspan") and v and v.isdigit())
            self.out.append(f"<td{spans}>")
            self.cell = []
        elif tag in ("table", "tr"):
            self.out.append(f"<{tag}>")
        elif tag == "br" and self.cell is not None:
            self.cell.append(" ")

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self.cell is not None:
            self.out.append(_inline("".join(self.cell).strip(), self.md) + "</td>")
            self.cell = None
        elif tag in ("table", "tr"):
            self.out.append(f"</{tag}>")

    def handle_data(self, data):
        if self.cell is not None:
            self.cell.append(data)


def _html_table(raw: str, md: bool) -> str:
    p = _TableCleaner(md)
    p.feed(raw)
    p.close()
    return "".join(p.out)


def _html(blocks: list[Block], md: bool) -> str:
    parts = []
    for b in blocks:
        if b.kind == "heading":
            parts.append(f"<p><b>{_inline(b.text, md)}</b></p>")
        elif b.kind == "para":
            parts.append(f"<p>{_inline(b.text, md)}</p>")
        elif b.kind == "list":
            marker = html.escape(b.prefix.strip())
            parts.append(f'<p style="margin-left: {b.level + 1}em; text-indent: -1em">{marker} {_inline(b.text, md)}</p>')
        elif b.kind == "table":
            parts.append(_table(b.rows, md))
        elif b.kind == "html_table":
            parts.append(_html_table(b.raw, md))
        elif b.kind in ("code", "math"):
            parts.append(f"<p>{html.escape(b.raw.strip().strip('`$'))}</p>")
        # raw blocks (image links, comments, rules) have nothing to draw
    return "\n".join(parts)


def _css(fonts: Fonts, size: float, leading: float, color, bold: bool, table: bool) -> str:
    pad = 1.0
    if table:  # measured spacing is the row pitch: give the difference to cell padding
        pad = max(1.0, (leading - 1.2) * size / 2)
        leading = 1.2
    return f"""{fonts.face_css}
* {{font-family: {fonts.family}; font-size: {size:.2f}px; line-height: {leading:.2f}; color: {_hex(color)};}}
body {{font-weight: {"bold" if bold else "normal"};}}
p {{margin: 0;}}
table {{border-collapse: collapse; width: 100%;}}
td {{border: 0.5px solid {_hex(color)}; padding: {pad:.1f}px 3px; vertical-align: middle;}}
"""


def _fit_leading(box: pymupdf.Rect, text: str, size: float, leading: float, color, bold: bool,
                 fonts: Fonts) -> str:
    """A longer translation first gives up line spacing, and only then font size."""
    best = None
    for lead in (leading, (leading + 1.2) / 2, 1.2):
        css = _css(fonts, size, lead, color, bold, False)
        probe = pymupdf.open()
        page = probe.new_page(width=box.x1 + 10, height=box.y1 + 10)
        _, scale = page.insert_htmlbox(box, text, css=css, archive=fonts.archive, scale_low=0)
        probe.close()
        if scale >= 0.999:
            return css
        if best is None or scale > best[0]:
            best = (scale, css)
    return best[1]


def render(doc: pymupdf.Document, regions: list[list[Region]], fn: Callable[[str], str],
           path: Path, fonts: Fonts) -> None:
    """Replace the text of every region with fn(text) and save the result to path."""
    for page, regs in zip(doc, regions):
        if not regs:
            continue
        pix = page.get_pixmap(dpi=_SAMPLE_DPI) if any(r.paint for r in regs) else None
        colors = [_sample(pix, r.rect) if r.paint else (None, r.color or (0, 0, 0)) for r in regs]
        translated = [markdown.map_text(r.blocks, fn) for r in regs]

        for r in regs:
            for e in r.erase:
                if not e.is_empty:
                    page.add_redact_annot(e, fill=False)
        if any(r.erase for r in regs):
            page.apply_redactions(images=pymupdf.PDF_REDACT_IMAGE_NONE,
                                  graphics=pymupdf.PDF_REDACT_LINE_ART_NONE)

        for r, (bg, fg), blocks in zip(regs, colors, translated):
            if not r.size and r.paint:
                r.size, r.leading = _ink_metrics(pix, r.rect, bg)
            size = r.size or _guess_size(r)
            box = r.rect
            if r.paint:
                page.draw_rect(r.rect + (-1, -1, 1, 1), color=None, fill=bg, width=0, overlay=True)
                box = _grow_blank(r, pix, bg, [o.rect for o in regs if o is not r], size)
            # a little room below: the box is the glyph bbox, the html layout needs full line height
            box = box + (0, 0, 0, 0.2 * size)
            text = _html(blocks, r.from_markdown)
            table = any(b.kind in ("table", "html_table") for b in blocks)
            css = _css(fonts, size, r.leading, fg, r.bold, table)
            if not table and r.leading > 1.3:
                css = _fit_leading(box, text, size, r.leading, fg, r.bold, fonts)
            page.insert_htmlbox(box, text, css=css, archive=fonts.archive, scale_low=0)
    try:
        doc.subset_fonts()  # otherwise each font is embedded whole, ~1 MB apiece
    except Exception as e:
        print(f"  ! шрифты не урезаны, PDF будет больше: {e}", flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(path, garbage=4, deflate=True)
