"""Deciding per page whether the PDF text layer is usable or the page needs OCR."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

import pymupdf
import pymupdf4llm

# Space that pymupdf inserts at line joins between two CJK characters
_CJK = r"　-〿぀-ヿ㐀-䶿一-鿿가-힯＀-￯"
_CJK_GAP = re.compile(rf"(?<=[{_CJK}]) (?=[{_CJK}])")


@dataclass
class PagePlan:
    index: int
    needs_ocr: bool
    markdown: str = ""  # filled for text-layer pages


def _text_layer_ok(text: str, min_chars: int) -> bool:
    chars = [c for c in text if not c.isspace()]
    if len(chars) < min_chars:
        return False
    # Broken font encodings show up as replacement chars, private-use glyphs or control chars
    bad = sum(
        1 for c in chars if c == "�" or unicodedata.category(c) in ("Co", "Cc", "Cn")
    )
    return bad / len(chars) < 0.05


def parse_pages(spec: str | None, count: int) -> list[int]:
    """'1-3,7' -> [0, 1, 2, 6] (0-based); None -> all pages."""
    if not spec:
        return list(range(count))
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a) - 1, min(int(b), count)))
        elif part:
            out.append(int(part) - 1)
    return [i for i in dict.fromkeys(out) if 0 <= i < count]


def plan_pages(doc: pymupdf.Document, pages: list[int], min_chars: int, force_ocr: bool) -> list[PagePlan]:
    plans = []
    for i in pages:
        if force_ocr or not _text_layer_ok(doc[i].get_text(), min_chars):
            plans.append(PagePlan(i, needs_ocr=True))
            continue
        chunk = pymupdf4llm.to_markdown(doc, pages=[i], page_chunks=True, show_progress=False)[0]
        md = _CJK_GAP.sub("", chunk["text"])
        plans.append(PagePlan(i, needs_ocr=False, markdown=md))
    return plans


def render_page_png(doc: pymupdf.Document, index: int, dpi: int) -> bytes:
    return doc[index].get_pixmap(dpi=dpi).tobytes("png")
