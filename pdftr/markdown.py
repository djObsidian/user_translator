"""Minimal markdown block model: parse OCR/text-layer markdown, map text through a translator, render back."""

from __future__ import annotations

import os
import re
import shutil
import unicodedata
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Callable

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_LIST = re.compile(r"^(\s*)([-*+•]|\d{1,3}[.)])\s+(.*)$")
_HR = re.compile(r"^\s*([-*_])(\s*\1){2,}\s*$")
# Whole-line image; the path is greedy because Windows paths may contain parentheses
_IMAGE = re.compile(r"^\s*!\[([^\]]*)\]\((.*)\)\s*$")
_TABLE_SEP = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$")
_HTML_CELL = re.compile(r"(<t[dh]\b[^>]*>)(.*?)(</t[dh]>)", re.S | re.I)
_CJK_CHAR = re.compile(r"[　-〿぀-ヿ㐀-䶿一-鿿가-힯＀-￯]")


@dataclass
class Block:
    kind: str  # heading | para | list | table | html_table | code | math | raw
    text: str = ""
    prefix: str = ""  # "## " for headings, "  - " for list items
    level: int = 0
    rows: list[list[str]] = field(default_factory=list)
    raw: str = ""


def image_path(line: str) -> Path | None:
    """Path of a whole-line markdown image, if the line is one."""
    m = _IMAGE.match(line)
    return Path(m.group(2).strip().strip("<>")) if m else None


def relink_images(md: str, md_dir: Path, img_dir: Path) -> str:
    """Copy images referenced by absolute paths into img_dir and point the links there, relative to md_dir."""

    def sub(line: str) -> str:
        src = image_path(line)
        if src is None or not src.is_absolute() or not src.is_file():
            return line
        img_dir.mkdir(parents=True, exist_ok=True)
        dst = img_dir / src.name
        shutil.copyfile(src, dst)
        rel = os.path.relpath(dst, md_dir).replace("\\", "/")
        return f"![{_IMAGE.match(line).group(1)}](<{rel}>)"

    return "\n".join(sub(line) for line in md.split("\n"))


def has_letters(text: str) -> bool:
    return any(unicodedata.category(c).startswith("L") for c in text)


def join_lines(lines: list[str]) -> str:
    """Join wrapped lines of one paragraph; no space between CJK characters."""
    out = ""
    for line in (l.strip() for l in lines):
        if not line:
            continue
        if out and not (_CJK_CHAR.match(out[-1]) and _CJK_CHAR.match(line[0])):
            out += " "
        out += line
    return out


def _split_row(line: str) -> list[str]:
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    return [c.strip() for c in re.split(r"(?<!\\)\|", line)]


def parse(md: str) -> list[Block]:
    lines = md.replace("\r\n", "\n").split("\n")
    blocks: list[Block] = []
    para: list[str] = []
    i = 0

    def flush_para():
        if para:
            blocks.append(Block("para", text=join_lines(para)))
            para.clear()

    while i < len(lines):
        line = lines[i]
        s = line.strip()

        if s.startswith("```") or s.startswith("~~~"):
            flush_para()
            fence, buf = s[:3], [line]
            i += 1
            while i < len(lines):
                buf.append(lines[i])
                if lines[i].strip().startswith(fence):
                    break
                i += 1
            blocks.append(Block("code", raw="\n".join(buf)))
        elif s.startswith("$$") or s.startswith("\\["):
            flush_para()
            end = "$$" if s.startswith("$$") else "\\]"
            buf = [line]
            closed = len(s) > 2 and s.endswith(end)
            while not closed and i + 1 < len(lines):
                i += 1
                buf.append(lines[i])
                closed = lines[i].strip().endswith(end)
            blocks.append(Block("math", raw="\n".join(buf)))
        elif s.lower().startswith("<table"):
            flush_para()
            buf = [line]
            while "</table>" not in lines[i].lower() and i + 1 < len(lines):
                i += 1
                buf.append(lines[i])
            blocks.append(Block("html_table", raw="\n".join(buf)))
        elif s.startswith("|"):
            flush_para()
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                if not _TABLE_SEP.match(lines[i]):
                    rows.append(_split_row(lines[i]))
                i += 1
            blocks.append(Block("table", rows=rows))
            continue
        elif not s:
            flush_para()
        elif m := _HEADING.match(s):
            flush_para()
            blocks.append(Block("heading", text=m.group(2).strip(" #"), prefix=m.group(1) + " ", level=len(m.group(1))))
        elif _HR.match(s) or _IMAGE.match(s) or s.startswith("<!--"):
            flush_para()
            blocks.append(Block("raw", raw=s))
        elif m := _LIST.match(line):
            flush_para()
            indent, marker, text = m.groups()
            item = [text]
            # continuation lines: indented, non-empty, not a new item
            while i + 1 < len(lines) and lines[i + 1].startswith((" ", "\t")) and lines[i + 1].strip() \
                    and not _LIST.match(lines[i + 1]):
                i += 1
                item.append(lines[i])
            blocks.append(Block("list", text=join_lines(item), prefix=f"{indent}{marker} ", level=len(indent) // 2))
        else:
            para.append(line)
        i += 1
    flush_para()
    return blocks


class _CellParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows: list[list[str]] = []
        self._cell: list[str] | None = None

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self.rows.append([])
        elif tag in ("td", "th"):
            if not self.rows:
                self.rows.append([])
            self._cell = []
        elif tag == "br" and self._cell is not None:
            self._cell.append(" ")

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self._cell is not None:
            self.rows[-1].append("".join(self._cell).strip())
            self._cell = None

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)


def html_table_rows(html: str) -> list[list[str]]:
    p = _CellParser()
    p.feed(html)
    return [r for r in p.rows if r]


def translatable_texts(blocks: list[Block]) -> list[str]:
    """Every distinct string the translator will be asked for, in document order."""
    out: list[str] = []
    for b in blocks:
        if b.kind in ("heading", "para", "list"):
            out.append(b.text)
        elif b.kind == "table":
            out.extend(c for r in b.rows for c in r)
        elif b.kind == "html_table":
            out.extend(m.group(2).strip() for m in _HTML_CELL.finditer(b.raw))
    return [t for t in dict.fromkeys(out) if has_letters(t)]


def map_text(blocks: list[Block], fn: Callable[[str], str]) -> list[Block]:
    """Return new blocks with every translatable string passed through fn."""

    def tr(t: str) -> str:
        return fn(t) if has_letters(t) else t

    out = []
    for b in blocks:
        if b.kind in ("heading", "para", "list"):
            out.append(Block(b.kind, text=tr(b.text), prefix=b.prefix, level=b.level))
        elif b.kind == "table":
            out.append(Block("table", rows=[[tr(c) for c in r] for r in b.rows]))
        elif b.kind == "html_table":
            raw = _HTML_CELL.sub(lambda m: m.group(1) + tr(m.group(2).strip()) + m.group(3), b.raw)
            out.append(Block("html_table", raw=raw))
        else:
            out.append(b)
    return out


def render(blocks: list[Block]) -> str:
    parts: list[str] = []
    prev = None
    for b in blocks:
        if b.kind in ("heading", "list"):
            s = b.prefix + b.text
        elif b.kind == "para":
            s = b.text
        elif b.kind == "table":
            if not b.rows:
                continue
            width = max(len(r) for r in b.rows)
            rows = [r + [""] * (width - len(r)) for r in b.rows]
            lines = ["| " + " | ".join(c.replace("|", "\\|") for c in r) + " |" for r in rows]
            lines.insert(1, "|" + "---|" * width)
            s = "\n".join(lines)
        else:
            s = b.raw
        # consecutive list items stay in one list
        sep = "\n" if prev == "list" and b.kind == "list" else "\n\n"
        parts.append((sep if parts else "") + s)
        prev = b.kind
    return "".join(parts) + "\n"
