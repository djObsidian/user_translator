"""Loading config.toml and resolving model files."""

from __future__ import annotations

import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config.toml"


@dataclass
class ModelProfile:
    name: str
    repo: str
    model: str
    mmproj: str | None = None
    chat_template: str | None = None
    prompt: str | None = None
    image_first: bool = False
    server_args: list[str] = field(default_factory=list)
    sampling: dict = field(default_factory=dict)

    def local_dir(self, models_dir: Path) -> Path:
        return models_dir / self.repo.replace("/", "__")

    def files(self) -> list[str]:
        return [f for f in (self.model, self.mmproj, self.chat_template) if f]


@dataclass
class Config:
    raw: dict
    base: Path

    @classmethod
    def load(cls, path: Path | None = None) -> "Config":
        path = Path(path) if path else DEFAULT_CONFIG
        with open(path, "rb") as f:
            return cls(tomllib.load(f), path.resolve().parent)

    def path(self, value: str) -> Path:
        p = Path(value)
        return p if p.is_absolute() else self.base / p

    @property
    def models_dir(self) -> Path:
        return self.path(self.raw["paths"]["models_dir"])

    @property
    def work_dir(self) -> Path:
        return self.path(self.raw["paths"]["work_dir"])

    @property
    def llama(self) -> dict:
        return self.raw["llama"]

    @property
    def ocr(self) -> dict:
        return self.raw["ocr"]

    @property
    def translate(self) -> dict:
        return self.raw["translate"]

    @property
    def output(self) -> dict:
        return self.raw.get("output", {})

    def llama_server_exe(self) -> Path:
        exe = "llama-server.exe" if sys.platform == "win32" else "llama-server"
        bin_dir = self.path(self.llama["bin_dir"])
        # Release archives sometimes unpack into a nested folder
        for cand in [bin_dir / exe, *bin_dir.glob(f"**/{exe}")]:
            if cand.is_file():
                return cand
        raise FileNotFoundError(
            f"llama-server не найден в {bin_dir}. Запустите setup.bat (или setup_models.py --llama)."
        )

    def profile(self, section: str, name: str | None = None) -> ModelProfile:
        sec = self.raw[section]
        name = name or sec["profile"]
        profiles = sec.get("profiles", {})
        if name not in profiles:
            raise KeyError(f"Профиль '{name}' не найден в [{section}.profiles]. Есть: {', '.join(profiles)}")
        p = profiles[name]
        return ModelProfile(
            name=name,
            repo=p["repo"],
            model=p["model"],
            mmproj=p.get("mmproj"),
            chat_template=p.get("chat_template"),
            prompt=p.get("prompt"),
            image_first=bool(p.get("image_first", False)),
            server_args=list(p.get("server_args", [])),
            sampling=dict(p.get("sampling", {})),
        )

    def model_path(self, profile: ModelProfile, filename: str | None) -> Path | None:
        if not filename:
            return None
        path = profile.local_dir(self.models_dir) / filename
        if not path.is_file():
            raise FileNotFoundError(
                f"Нет файла модели {path}. Скачайте: setup_models.py --ocr/--translate {profile.name}"
            )
        return path
