"""CLI: python -m pdftr file.pdf [...] --to ru"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf
from tqdm import tqdm

from . import langs, markdown
from .config import Config
from .docx_out import write_docx
from .ingest import PagePlan, parse_pages, plan_pages, render_page_png
from .ocr import ocr_page
from .server import LlamaServer
from .translate import Cache, Translator, load_glossary


@dataclass
class Job:
    pdf: Path
    doc: pymupdf.Document
    work: Path
    plans: list[PagePlan]
    page_md: dict[int, str] = field(default_factory=dict)


def _file_hash(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:10]


def _elapsed(t0: float) -> str:
    s = int(time.monotonic() - t0)
    return f"{s // 3600}:{s // 60 % 60:02d}:{s % 60:02d}"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="pdftr", description="Офлайн-перевод PDF: OCR (при необходимости) + перевод в Markdown/DOCX.")
    ap.add_argument("pdfs", nargs="+", type=Path, help="PDF-файлы")
    ap.add_argument("--to", dest="target", help="целевой язык: ru, en, zh, de, ... (по умолчанию из config.toml)")
    ap.add_argument("-o", "--out", type=Path, help="папка для результатов (по умолчанию рядом с PDF)")
    ap.add_argument("--pages", help="диапазон страниц, например 1-10,15")
    ap.add_argument("--force-ocr", action="store_true", help="распознавать все страницы, даже с текстовым слоем")
    ap.add_argument("--ocr-only", action="store_true", help="только распознать в <name>.src.md, не переводить")
    ap.add_argument("--ocr-profile", help="профиль OCR из config.toml")
    ap.add_argument("--mt-profile", help="профиль перевода из config.toml")
    ap.add_argument("--glossary", type=Path, help="файл терминов: строки 'исходное = перевод'")
    ap.add_argument("--dpi", type=int, help="DPI рендера страниц для OCR")
    ap.add_argument("--no-page-breaks", action="store_true", help="не ставить разрывы страниц в DOCX")
    ap.add_argument("--config", type=Path, help="путь к config.toml")
    return ap


def stage_plan(cfg: Config, args) -> list[Job]:
    jobs = []
    for pdf in args.pdfs:
        if not pdf.is_file():
            print(f"[skip] нет файла: {pdf}")
            continue
        doc = pymupdf.open(pdf)
        if doc.needs_pass:
            print(f"[skip] PDF защищён паролем: {pdf}")
            continue
        pages = parse_pages(args.pages, doc.page_count)
        plans = plan_pages(doc, pages, int(cfg.ocr["min_text_chars"]), args.force_ocr)
        n_ocr = sum(p.needs_ocr for p in plans)
        print(f"[plan] {pdf.name}: {len(plans)} стр., текстовый слой: {len(plans) - n_ocr}, OCR: {n_ocr}")
        job = Job(pdf, doc, cfg.work_dir / f"{pdf.stem}-{_file_hash(pdf)}", plans)
        for p in plans:
            if not p.needs_ocr:
                job.page_md[p.index] = p.markdown
        jobs.append(job)
    return jobs


def stage_ocr(cfg: Config, args, jobs: list[Job]) -> None:
    profile = cfg.profile("ocr", args.ocr_profile)
    dpi = args.dpi or int(cfg.ocr["dpi"])
    pending = []
    for job in jobs:
        for p in job.plans:
            if not p.needs_ocr:
                continue
            cache = job.work / "pages" / f"{p.index + 1:04d}.{profile.name}.{dpi}.md"
            if cache.is_file():
                job.page_md[p.index] = cache.read_text(encoding="utf-8")
            else:
                pending.append((job, p, cache))
    if not pending:
        return
    t0 = time.monotonic()
    log = cfg.work_dir / "llama-ocr.log"
    with LlamaServer(cfg, profile, int(cfg.ocr["ctx"]), log) as srv:
        for job, p, cache in tqdm(pending, desc="OCR", unit="стр"):
            png = render_page_png(job.doc, p.index, dpi)
            md = ocr_page(srv, png, int(cfg.ocr["max_tokens"]))
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(md, encoding="utf-8")
            job.page_md[p.index] = md
    print(f"[ocr] {len(pending)} стр. за {_elapsed(t0)}")


def _page_marker(index: int) -> str:
    return f"<!-- page {index + 1} -->"


def write_source(job: Job, out_dir: Path) -> Path:
    path = out_dir / f"{job.pdf.stem}.src.md"
    parts = [f"{_page_marker(p.index)}\n\n{job.page_md[p.index].strip()}\n" for p in job.plans]
    path.write_text("\n".join(parts), encoding="utf-8")
    return path


def stage_translate(cfg: Config, args, jobs: list[Job], out_dirs: dict[Path, Path]) -> None:
    code, name = langs.resolve(args.target or cfg.translate["target"])
    profile = cfg.profile("translate", args.mt_profile)
    cache = Cache(cfg.work_dir / "translations.sqlite")
    tr = Translator(
        None, cache,
        target_code=code, target_name=name, model_id=profile.repo + "/" + profile.model,
        sampling=profile.sampling, max_tokens=int(cfg.translate["max_tokens"]),
        chunk_chars=int(cfg.translate["chunk_chars"]), glossary=load_glossary(args.glossary),
    )
    parsed = {id(job): [markdown.parse(job.page_md[p.index]) for p in job.plans] for job in jobs}
    todo = list(dict.fromkeys(
        t for job in jobs for blocks in parsed[id(job)] for t in markdown.translatable_texts(blocks)
        if not tr.is_cached(t)
    ))
    todo_set = set(todo)

    def run_all():
        bar = tqdm(total=len(todo), desc=f"Перевод -> {name}", unit="фрагм")
        seen: set[str] = set()

        def fn(text: str) -> str:
            out = tr(text)
            if text not in seen and text in todo_set:
                seen.add(text)
                bar.update(1)
            return out

        for job in jobs:
            pages = [markdown.map_text(blocks, fn) for blocks in parsed[id(job)]]
            out_dir = out_dirs[job.pdf]
            md_path = out_dir / f"{job.pdf.stem}.{code}.md"
            md_path.write_text(
                "\n".join(f"{_page_marker(p.index)}\n\n{markdown.render(b)}" for p, b in zip(job.plans, pages)),
                encoding="utf-8",
            )
            docx_path = out_dir / f"{job.pdf.stem}.{code}.docx"
            write_docx(pages, docx_path, page_breaks=not args.no_page_breaks)
            tqdm.write(f"[done] {md_path}\n[done] {docx_path}")
        bar.close()

    t0 = time.monotonic()
    if todo:
        print(f"[translate] фрагментов к переводу: {len(todo)} (остальное из кэша)")
        with LlamaServer(cfg, profile, int(cfg.translate["ctx"]), cfg.work_dir / "llama-mt.log") as srv:
            tr.server = srv
            run_all()
        print(f"[translate] готово за {_elapsed(t0)}")
    else:
        run_all()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = Config.load(args.config)
    t0 = time.monotonic()

    jobs = stage_plan(cfg, args)
    if not jobs:
        return 1
    out_dirs = {}
    for job in jobs:
        out_dirs[job.pdf] = args.out or job.pdf.resolve().parent
        out_dirs[job.pdf].mkdir(parents=True, exist_ok=True)

    stage_ocr(cfg, args, jobs)
    for job in jobs:
        print(f"[src] {write_source(job, out_dirs[job.pdf])}")
    if not args.ocr_only:
        stage_translate(cfg, args, jobs, out_dirs)
    print(f"Всё готово за {_elapsed(t0)}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nПрервано. Уже сделанное сохранено в кэше, повторный запуск продолжит с того же места.")
        sys.exit(130)
