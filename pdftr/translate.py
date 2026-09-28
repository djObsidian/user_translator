"""Text translation with Hy-MT2 via llama-server, with a sqlite cache so interrupted runs resume."""

from __future__ import annotations

import hashlib
import re
import sqlite3
from pathlib import Path

from .langs import NO_SPACE
from .server import LlamaServer

_SENTENCE_END = re.compile(r"(?<=[。！？!?；;])|(?<=[.:])(?=\s)")


def split_long(text: str, limit: int) -> list[str]:
    """Split text longer than `limit` chars at sentence boundaries."""
    if len(text) <= limit:
        return [text]
    pieces, cur = [], ""
    for sent in (s for s in _SENTENCE_END.split(text) if s):
        if cur and len(cur) + len(sent) > limit:
            pieces.append(cur.strip())
            cur = ""
        while len(sent) > limit:  # a single monster "sentence" (e.g. no punctuation at all)
            pieces.append(sent[:limit].strip())
            sent = sent[limit:]
        cur += sent
    if cur.strip():
        pieces.append(cur.strip())
    return pieces


def load_glossary(path: Path | None) -> dict[str, str]:
    """Lines 'source = target' (or tab-separated). '#' starts a comment."""
    if not path:
        return {}
    out = {}
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        src, _, tgt = line.partition("\t") if "\t" in line else line.partition("=")
        if src.strip() and tgt.strip():
            out[src.strip()] = tgt.strip()
    return out


def build_prompt(text: str, target_name: str, terms: dict[str, str]) -> str:
    if terms:
        refs = "\n".join(f"{s} translates to {t}" for s, t in terms.items())
        return (
            f"Reference the following translations:\n{refs}\n\n"
            f"Translate the following text into {target_name}. Note that you must ONLY output "
            f"the translated result without any additional explanation:\n\n{text}"
        )
    return (
        f"Translate the following text into {target_name}. Note that you should only output "
        f"the translated result without any additional explanation:\n\n{text}"
    )


class Cache:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute("CREATE TABLE IF NOT EXISTS tr (key TEXT PRIMARY KEY, out TEXT NOT NULL)")

    def get(self, key: str) -> str | None:
        row = self.db.execute("SELECT out FROM tr WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def put(self, key: str, out: str) -> None:
        self.db.execute("INSERT OR REPLACE INTO tr VALUES (?, ?)", (key, out))
        self.db.commit()


class Translator:
    def __init__(self, server: LlamaServer | None, cache: Cache, *, target_code: str, target_name: str,
                 model_id: str, sampling: dict, max_tokens: int, chunk_chars: int,
                 glossary: dict[str, str]):
        self.server = server
        self.cache = cache
        self.target_code = target_code
        self.target_name = target_name
        self.model_id = model_id
        self.sampling = sampling
        self.max_tokens = max_tokens
        self.chunk_chars = chunk_chars
        self.glossary = glossary

    def _key(self, text: str, terms: dict[str, str]) -> str:
        h = hashlib.sha256()
        for part in (self.model_id, self.target_name, repr(sorted(terms.items())), text):
            h.update(part.encode("utf-8") + b"\0")
        return h.hexdigest()

    def _terms_for(self, text: str) -> dict[str, str]:
        return {s: t for s, t in self.glossary.items() if s in text}

    def is_cached(self, text: str) -> bool:
        return all(self.cache.get(self._key(p, self._terms_for(p))) is not None
                   for p in split_long(text, self.chunk_chars))

    def __call__(self, text: str) -> str:
        sep = "" if self.target_code in NO_SPACE else " "
        return sep.join(self._one(p) for p in split_long(text, self.chunk_chars))

    def _one(self, text: str) -> str:
        terms = self._terms_for(text)
        key = self._key(text, terms)
        if (hit := self.cache.get(key)) is not None:
            return hit
        if self.server is None:
            raise RuntimeError("перевод не в кэше, а сервер не запущен")
        prompt = build_prompt(text, self.target_name, terms)
        out, finish = self.server.chat(prompt, max_tokens=self.max_tokens, **self.sampling)
        if finish == "length" or not out.strip():
            # Most likely a repetition loop: retry once, colder and with a repetition penalty
            out, finish = self.server.chat(prompt, max_tokens=self.max_tokens,
                                           **{**self.sampling, "temperature": 0.2, "repeat_penalty": 1.1})
            if finish == "length":
                print(f"  ! перевод обрезан по max_tokens: {text[:60]!r}...", flush=True)
        out = re.sub(r"<think>.*?</think>", "", out, flags=re.S).strip()
        self.cache.put(key, out)
        return out
