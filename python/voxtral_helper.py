#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import sys
import time
from pathlib import Path
import stat
from typing import Any

VOICES = [
    "casual_male",
    "casual_female",
    "cheerful_female",
    "neutral_male",
    "neutral_female",
    "fr_male",
    "fr_female",
    "es_male",
    "es_female",
    "de_male",
    "de_female",
    "it_male",
    "it_female",
    "pt_male",
    "pt_female",
    "nl_male",
    "nl_female",
    "ar_male",
    "hi_male",
    "hi_female",
]
SAMPLE_RATE = 24_000
TARGET_PEAK = 0.9
MIN_AUDIO_PEAK = 1e-6
PROFILE_TO_DENOISING_STEPS = {
    "quality": 8,
    "balanced": 6,
    "fast": 4,
}
DEFAULT_PROFILE = "balanced"
DEFAULT_VOICE = "casual_female"
MAX_TEXT_CHARS = 10_000
ALLOWED_FORMATS = {"wav", "flac", "mp3"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download")
    download.add_argument("--model-repo", required=True)
    download.add_argument("--model-dir", required=True)

    daemon = subparsers.add_parser("daemon")
    daemon.add_argument("--socket", required=True)
    daemon.add_argument("--pid-file", required=True)
    daemon.add_argument("--model-repo", required=True)
    daemon.add_argument("--model-dir", required=True)

    return parser.parse_args()


def do_download(model_repo: str, model_dir: Path) -> None:
    from huggingface_hub import snapshot_download

    model_dir.parent.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=model_repo, local_dir=str(model_dir))


class VoxtralDaemon:
    def __init__(self, socket_path: Path, pid_file: Path, model_repo: str, model_dir: Path) -> None:
        self.socket_path = socket_path
        self.pid_file = pid_file
        self.model_repo = model_repo
        self.model_dir = model_dir
        self.server: asyncio.base_events.Server | None = None
        self.model = None
        self.lock = asyncio.Lock()
        self.shutting_down = False

    async def start(self) -> None:
        from mlx_audio.tts.utils import load

        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        self.pid_file.parent.mkdir(parents=True, exist_ok=True)
        if self.socket_path.exists():
            self.socket_path.unlink()
        self.pid_file.write_text(str(os.getpid()))
        self.model = load(str(self.model_dir))
        self._apply_profile(DEFAULT_PROFILE)
        self._warm_voice_embedding(DEFAULT_VOICE)
        await self._warmup()
        self.server = await asyncio.start_unix_server(self._handle_client, path=str(self.socket_path))
        os.chmod(self.socket_path, stat.S_IRUSR | stat.S_IWUSR)
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, lambda s=sig: asyncio.create_task(self.shutdown()))
        async with self.server:
            try:
                await self.server.serve_forever()
            except asyncio.CancelledError:
                pass

    async def shutdown(self) -> None:
        if self.shutting_down:
            return
        self.shutting_down = True
        if self.server is not None:
            self.server.close()
            await self.server.wait_closed()
        if self.socket_path.exists():
            self.socket_path.unlink()
        if self.pid_file.exists():
            self.pid_file.unlink()

    async def _warmup(self) -> None:
        output = self.model_dir.parent / "warmup.wav"
        try:
            await self._speak("Warm up.", DEFAULT_VOICE, DEFAULT_PROFILE, output, "wav")
        finally:
            if output.exists():
                output.unlink()

    async def _handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        line = await reader.readline()
        if not line:
            writer.close()
            await writer.wait_closed()
            return
        try:
            request = json.loads(line)
            response = await self._dispatch(request)
        except Exception as exc:  # noqa: BLE001
            response = {
                "ok": False,
                "error": str(exc),
                "backend": "mlx_daemon",
                "model": self.model_repo,
            }
        writer.write(json.dumps(response).encode("utf-8") + b"\n")
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    async def _dispatch(self, request: dict[str, Any]) -> dict[str, Any]:
        kind = request.get("kind")
        if kind == "ping":
            return {
                "ok": True,
                "backend": "mlx_daemon",
                "model": self.model_repo,
            }
        if kind == "shutdown":
            asyncio.create_task(self.shutdown())
            return {"ok": True, "backend": "mlx_daemon", "model": self.model_repo}
        if kind == "list_voices":
            return {
                "ok": True,
                "backend": "mlx_daemon",
                "model": self.model_repo,
                "voices": VOICES,
            }
        if kind == "speak":
            text = request.get("text")
            voice = request.get("voice")
            profile = request.get("profile", DEFAULT_PROFILE)
            output_path_raw = request.get("output_path")
            if not isinstance(text, str) or not text.strip():
                raise ValueError("text is required")
            if len(text) > MAX_TEXT_CHARS:
                raise ValueError(f"text exceeds max length of {MAX_TEXT_CHARS} characters")
            if not isinstance(voice, str) or voice not in VOICES:
                raise ValueError("voice must be a supported preset")
            if not isinstance(output_path_raw, str) or not output_path_raw:
                raise ValueError("output_path is required")
            output_path = Path(output_path_raw)
            if not output_path.is_absolute():
                raise ValueError("output_path must be an absolute path")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fmt = request.get("format", "wav")
            if fmt not in ALLOWED_FORMATS:
                raise ValueError(f"unsupported format: {fmt}")
            queue_start = time.perf_counter()
            async with self.lock:
                queue_ms = (time.perf_counter() - queue_start) * 1000.0
                audio_seconds, wall_ms = await self._speak(text, voice, profile, output_path, fmt)
            return {
                "ok": True,
                "backend": "mlx_daemon",
                "model": self.model_repo,
                "output_path": str(output_path),
                "audio_seconds": audio_seconds,
                "wall_ms": wall_ms,
                "queue_ms": queue_ms,
            }
        raise ValueError(f"unknown request kind: {kind}")

    async def _speak(
        self, text: str, voice: str, profile: str, output_path: Path, fmt: str
    ) -> tuple[float, float]:
        import numpy as np
        import soundfile as sf

        self._apply_profile(profile)
        self._warm_voice_embedding(voice)
        start = time.perf_counter()
        results = list(self.model.generate(text=text, voice=voice))
        if not results:
            raise RuntimeError("model returned no audio")
        result = results[-1]
        audio = normalize_audio(np.asarray(result.audio))
        sample_rate = int(getattr(result, "sample_rate", SAMPLE_RATE))
        subtype = None
        if fmt == "wav":
            subtype = "PCM_16"
        sf.write(str(output_path), audio, sample_rate, subtype=subtype)
        wall_ms = (time.perf_counter() - start) * 1000.0
        audio_seconds = float(audio.shape[0]) / float(sample_rate)
        return audio_seconds, wall_ms

    def _apply_profile(self, profile: str) -> None:
        args = getattr(self.model.acoustic_transformer, "args", None)
        if args is None:
            raise RuntimeError("acoustic transformer args not available")
        args.n_denoising_steps = PROFILE_TO_DENOISING_STEPS.get(
            profile, PROFILE_TO_DENOISING_STEPS[DEFAULT_PROFILE]
        )

    def _warm_voice_embedding(self, voice: str) -> None:
        get_voice_embedding = getattr(self.model, "_get_voice_embedding", None)
        if get_voice_embedding is None:
            return
        get_voice_embedding(voice)


def normalize_audio(audio):
    import numpy as np

    audio = np.asarray(audio, dtype=np.float32)
    if not audio.size:
        return audio
    peak = float(np.max(np.abs(audio)))
    if peak <= MIN_AUDIO_PEAK:
        return audio
    gain = min(TARGET_PEAK / peak, 8.0)
    normalized = np.clip(audio * gain, -1.0, 1.0)
    return normalized


async def run_daemon(args: argparse.Namespace) -> None:
    daemon = VoxtralDaemon(
        socket_path=Path(args.socket),
        pid_file=Path(args.pid_file),
        model_repo=args.model_repo,
        model_dir=Path(args.model_dir),
    )
    try:
        await daemon.start()
    finally:
        await daemon.shutdown()


def main() -> int:
    args = parse_args()
    if args.command == "download":
        do_download(args.model_repo, Path(args.model_dir))
        return 0
    if args.command == "daemon":
        asyncio.run(run_daemon(args))
        return 0
    raise ValueError(f"unsupported command {args.command}")


if __name__ == "__main__":
    sys.exit(main())
