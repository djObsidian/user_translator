"""Page image -> markdown via an OCR VLM served by llama-server."""

from __future__ import annotations

import base64
import re

from .server import LlamaServer

# DeepSeek-OCR grounding output: <|ref|>label<|/ref|><|det|>[[x1,y1,x2,y2]]<|/det|>
_GROUNDING = re.compile(r"<\|ref\|>.*?<\|/ref\|>\s*<\|det\|>.*?<\|/det\|>[ \t]*\n?", re.S)
# <|...|> and DeepSeek's fullwidth <｜end▁of▁sentence｜>
_SPECIAL = re.compile(r"<[|｜][^|｜>]{1,40}[|｜]>")
_FENCED = re.compile(r"^\s*```(?:markdown|md)?\s*\n(.*)\n\s*```\s*$", re.S)


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


def clean_ocr_output(text: str) -> str:
    text = _GROUNDING.sub("", text)
    text = _SPECIAL.sub("", text)
    m = _FENCED.match(text)
    if m:
        text = m.group(1)
    text = _collapse_loops(text)
    return re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"


def ocr_page(server: LlamaServer, png: bytes, max_tokens: int) -> str:
    image = {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode()}}
    prompt = {"type": "text", "text": server.profile.prompt or "OCR"}
    content = [image, prompt] if server.profile.image_first else [prompt, image]
    text, finish = server.chat(content, max_tokens=max_tokens, temperature=0.0, top_k=1)
    if finish == "length":
        print("  ! OCR упёрся в max_tokens, страница могла обрезаться", flush=True)
    return clean_ocr_output(text)
