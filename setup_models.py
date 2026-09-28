"""Download llama.cpp (CPU build) and GGUF models. Needs internet once; after that everything runs offline.

    python setup_models.py                      # llama.cpp + models from config.toml profiles
    python setup_models.py --translate hy-mt2-7b --ocr glm-ocr
    python setup_models.py --llama              # only (re)download llama.cpp
"""

from __future__ import annotations

import argparse
import io
import json
import platform
import re
import shutil
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

from pdftr.config import Config

RELEASES_API = "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=30"


def _asset_pattern() -> re.Pattern:
    arch = "arm64" if platform.machine().lower() in ("arm64", "aarch64") else "x64"
    if sys.platform == "win32":
        return re.compile(rf"^llama-b\d+-bin-win-cpu-{arch}\.zip$")
    if sys.platform == "darwin":
        return re.compile(rf"^llama-b\d+-bin-macos-{arch}\.tar\.gz$")
    return re.compile(rf"^llama-b\d+-bin-ubuntu-{arch}\.tar\.gz$")


def _download(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as r:
        total = int(r.headers.get("Content-Length") or 0)
        buf = io.BytesIO()
        while chunk := r.read(1 << 20):
            buf.write(chunk)
            if total:
                print(f"\r  {buf.tell() / 1e6:.0f}/{total / 1e6:.0f} MB", end="", flush=True)
        print()
        return buf.getvalue()


def install_llama(cfg: Config) -> None:
    pattern = _asset_pattern()
    with urllib.request.urlopen(RELEASES_API, timeout=60) as r:
        releases = json.load(r)
    for rel in releases:
        asset = next((a for a in rel["assets"] if pattern.match(a["name"])), None)
        if asset:
            break
    else:
        sys.exit(f"Не нашёл сборку llama.cpp под эту платформу ({pattern.pattern}). Соберите вручную и укажите bin_dir.")

    dest = cfg.path(cfg.llama["bin_dir"])
    print(f"[llama.cpp] {rel['tag_name']}: {asset['name']}")
    data = _download(asset["browser_download_url"])
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    if asset["name"].endswith(".zip"):
        zipfile.ZipFile(io.BytesIO(data)).extractall(dest)
    else:
        tarfile.open(fileobj=io.BytesIO(data)).extractall(dest, filter="data")
        for exe in dest.glob("**/llama-*"):
            exe.chmod(0o755)
    (dest / "VERSION").write_text(rel["tag_name"])
    print(f"[llama.cpp] установлен в {dest}")


def install_model(cfg: Config, section: str, name: str | None) -> None:
    from huggingface_hub import hf_hub_download

    profile = cfg.profile(section, name)
    local = profile.local_dir(cfg.models_dir)
    for filename in profile.files():
        if (local / filename).is_file():
            print(f"[{section}] {profile.name}: {filename} уже скачан")
            continue
        print(f"[{section}] {profile.name}: качаю {profile.repo}/{filename}")
        hf_hub_download(profile.repo, filename, local_dir=local)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llama", action="store_true", help="только (пере)скачать llama.cpp")
    ap.add_argument("--ocr", nargs="*", help="профили OCR (по умолчанию из config.toml)")
    ap.add_argument("--translate", nargs="*", help="профили перевода (по умолчанию из config.toml)")
    ap.add_argument("--config", type=Path)
    args = ap.parse_args()
    cfg = Config.load(args.config)

    if args.llama:
        install_llama(cfg)
        return
    try:
        cfg.llama_server_exe()
    except FileNotFoundError:
        install_llama(cfg)
    for name in args.ocr or [None]:
        install_model(cfg, "ocr", name)
    for name in args.translate or [None]:
        install_model(cfg, "translate", name)
    print("\nГотово. Дальше всё работает без интернета.")


if __name__ == "__main__":
    main()
