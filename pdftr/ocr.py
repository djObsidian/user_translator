"""Page image -> markdown via an OCR VLM served by llama-server."""

from __future__ import annotations

import base64
import re
from dataclasses import dataclass
from pathlib import Path

import pymupdf

from .server import LlamaServer

# DeepSeek-OCR grounding output: <|ref|>label<|/ref|><|det|>[[x1,y1,x2,y2]]<|/det|>, coords 0..999
_REGION = re.compile(r"<\|ref\|>(.*?)<\|/ref\|>\s*<\|det\|>(.*?)<\|/det\|>[ \t]*\n?", re.S)
_GRID = 999
# <|...|> and DeepSeek's fullwidth <｜end▁of▁sentence｜>
_SPECIAL = re.compile(r"<[|｜][^|｜>]{1,40}[|｜]>")
_TABLE_START = re.compile(r"<table\b|^\s*\|", re.I | re.M)
_FENCED = re.compile(r"^\s*```(?:markdown|md)?\s*\n(.*)\n\s*```\s*$", re.S)

# Region labels that are pictures (kept as is) and formulas (not translated)
IMAGE_LABELS = {"image", "figure", "picture", "chart", "photo", "diagram", "seal", "logo"}
FORMULA_LABELS = {"equation", "formula", "isolate_formula", "interline_equation"}
TITLE_LABELS = {"title", "sub_title", "subtitle", "header_title", "doc_title", "paragraph_title"}


@dataclass
class Region:
    label: str
    box: tuple[float, float, float, float]  # x0, y0, x1, y1 as fractions of the page
    content: str


def _collapse_loops(text: str, max_repeats: int = 3) -> str:
    """Small VLMs sometimes get stuck repeating one line until max_tokens."""
    out: list[str] = []
    run = 0
    for line in text.split("\n"):
        if out and line.strip() and line == out[-1]:
            run += 1
            if run >= max_repeats:
                continue
        else:
            run = 0
        out.append(line)
    return "\n".join(out)


def _unfence(text: str) -> str:
    m = _FENCED.match(text)
    return m.group(1) if m else text


def _clean(text: str) -> str:
    text = _collapse_loops(_SPECIAL.sub("", text))
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def clean_ocr_output(text: str) -> str:
    return _clean(_REGION.sub("", _unfence(text))) + "\n"


def parse_regions(raw: str) -> list[Region]:
    """Layout regions with page coordinates; empty if the model gave no grounding boxes."""
    raw = _unfence(raw)
    matches = list(_REGION.finditer(raw))
    out = []
    for m, nxt in zip(matches, matches[1:] + [None]):
        nums = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", m.group(2))]
        boxes = [nums[k:k + 4] for k in range(0, len(nums) - 3, 4)]
        if not boxes:
            continue
        box = tuple(
            min(max(v / _GRID, 0.0), 1.0)
            for v in (min(b[0] for b in boxes), min(b[1] for b in boxes),
                      max(b[2] for b in boxes), max(b[3] for b in boxes))
        )
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        content = _clean(raw[m.end():nxt.start() if nxt else len(raw)])
        out.append(Region(m.group(1).strip().lower(), box, content))
    # DeepSeek-OCR sometimes emits <table> tag, then <table_caption> tag, then caption text and the table:
    # the table belongs to the empty table region, not to the small caption box
    for cur, nxt in zip(out, out[1:]):
        if cur.label == "table" and not cur.content and (m := _TABLE_START.search(nxt.content)):
            cur.content = nxt.content[m.start():].strip()
            nxt.content = nxt.content[:m.start()].strip()
    return out


def save_crops(png: bytes, crops: dict[Path, tuple[float, float, float, float]], dpi: int) -> None:
    """Cut picture regions out of the rendered page."""
    page = pymupdf.Pixmap(png)
    for path, (x0, y0, x1, y1) in crops.items():
        ir = pymupdf.IRect(x0 * page.width, y0 * page.height, x1 * page.width, y1 * page.height) & page.irect
        if ir.is_empty:
            continue
        pix = pymupdf.Pixmap(page.colorspace, ir, page.alpha)
        pix.copy(page, ir)
        pix.set_dpi(dpi, dpi)
        path.parent.mkdir(parents=True, exist_ok=True)
        pix.save(path)


def to_markdown(raw: str, regions: list[Region], images: dict[int, Path]) -> str:
    """Markdown of the page; picture regions become links to their crops in `images`."""
    if not regions:
        return clean_ocr_output(raw)
    parts = []
    for i, r in enumerate(regions):
        if i in images:
            parts.append(f"![]({images[i]})")
        elif r.content:
            parts.append(r.content)
    return "\n\n".join(parts) + "\n"


def ocr_page(server: LlamaServer, png: bytes, max_tokens: int) -> str:
    """Raw model output, grounding tags included (they carry the layout)."""
    image = {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode()}}
    prompt = {"type": "text", "text": server.profile.prompt or "OCR"}
    content = [image, prompt] if server.profile.image_first else [prompt, image]
    text, finish = server.chat(content, max_tokens=max_tokens, temperature=0.0, top_k=1)
    if finish == "length":
        print("  ! OCR упёрся в max_tokens, страница могла обрезаться", flush=True)
    return text
