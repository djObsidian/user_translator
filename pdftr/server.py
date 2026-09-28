"""Running llama-server as a subprocess and talking to its OpenAI-compatible API."""

from __future__ import annotations

import json
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from .config import Config, ModelProfile


def _port_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((host, port)) != 0


def _pick_port(host: str, preferred: int) -> int:
    if _port_free(host, preferred):
        return preferred
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return s.getsockname()[1]


class LlamaServer:
    """Context manager: starts llama-server with one model, stops it on exit."""

    def __init__(self, cfg: Config, profile: ModelProfile, ctx: int, log_path: Path):
        self.cfg = cfg
        self.profile = profile
        self.ctx = ctx
        self.log_path = log_path
        self.host = cfg.llama["host"]
        self.port = _pick_port(self.host, int(cfg.llama["port"]))
        self.proc: subprocess.Popen | None = None
        self._log = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def _cmd(self) -> list[str]:
        cfg, p = self.cfg, self.profile
        cmd = [
            str(cfg.llama_server_exe()),
            "-m", str(cfg.model_path(p, p.model)),
            "--host", self.host,
            "--port", str(self.port),
            "-c", str(self.ctx),
            "-np", "1",
            "--no-webui",
        ]
        if p.mmproj:
            cmd += ["--mmproj", str(cfg.model_path(p, p.mmproj))]
        if p.chat_template:
            cmd += ["--jinja", "--chat-template-file", str(cfg.model_path(p, p.chat_template))]
        threads = int(cfg.llama.get("threads", 0))
        if threads > 0:
            cmd += ["-t", str(threads)]
        return cmd + p.server_args

    def __enter__(self) -> "LlamaServer":
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = open(self.log_path, "ab")
        cmd = self._cmd()
        self._log.write(("\n\n$ " + " ".join(cmd) + "\n").encode())
        self._log.flush()
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        self.proc = subprocess.Popen(cmd, stdout=self._log, stderr=subprocess.STDOUT, creationflags=flags)
        print(f"[llama] запускаю {self.profile.name} ...", flush=True)
        self._wait_ready()
        return self

    def _wait_ready(self, timeout: float = 900) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(
                    f"llama-server упал при старте (код {self.proc.returncode}). Лог: {self.log_path}"
                )
            try:
                with urllib.request.urlopen(self.base_url + "/health", timeout=5) as r:
                    if r.status == 200:
                        print(f"[llama] {self.profile.name} готов", flush=True)
                        return
            except (urllib.error.URLError, ConnectionError, TimeoutError):
                pass
            time.sleep(1)
        raise TimeoutError(f"llama-server не поднялся за {timeout:.0f} с. Лог: {self.log_path}")

    def __exit__(self, *exc) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        if self._log:
            self._log.close()

    def chat(self, content, *, max_tokens: int, **sampling) -> tuple[str, str]:
        """One user turn -> (text, finish_reason). `content` is a str or a list of content parts."""
        payload = {
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens,
            "stream": False,
            **sampling,
        }
        req = urllib.request.Request(
            self.base_url + "/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        # CPU inference can be very slow; do not give up on a single long page
        with urllib.request.urlopen(req, timeout=4 * 3600) as r:
            data = json.loads(r.read().decode("utf-8"))
        choice = data["choices"][0]
        return choice["message"].get("content") or "", choice.get("finish_reason") or ""
