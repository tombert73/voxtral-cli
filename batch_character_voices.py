#!/usr/bin/env python3
"""
batch_character_voices.py

Batch-generate spoken lines for fiction characters on Apple Silicon, cloning a
distinct accent per character from a short reference clip.

Pipeline: mlx-audio + LongCat-AudioDiT (zero-shot voice cloning, 24 kHz, MIT).
Each character points at one reference WAV (e.g. a trimmed Speech Accent Archive
sample) plus the transcript of exactly what is said in that clip. Every line for
that character reuses the same reference, so the voice stays consistent.

Setup (run once in a venv):
    pip install mlx-audio soundfile
    # WAV output needs no ffmpeg; install it only if you later want mp3/flac:
    # brew install ffmpeg

Run:
    python batch_character_voices.py

Swapping models: change MODEL below. LongCat's generate() signature is filled in
here. Other cloning models in mlx-audio (Chatterbox, Higgs Audio v2, CSM, Ming
Omni) use the same load()/ref_audio idea but slightly different generate kwargs,
so check that model's README before swapping and adjust synth() accordingly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

# --------------------------------------------------------------------------- #
# CONFIG
# --------------------------------------------------------------------------- #

MODEL = "mlx-community/LongCat-AudioDiT-1B-bf16"
SAMPLE_RATE = 24000  # LongCat-AudioDiT outputs 24 kHz

REF_DIR = Path("refs")        # where your trimmed reference clips live
OUTPUT_DIR = Path("out")      # where generated lines are written

# If True, ignore the ref_text you typed and auto-transcribe each reference clip
# with the Whisper model bundled in mlx-audio. Safer when you have trimmed a clip
# and are not 100% sure of the exact words remaining in it.
AUTO_TRANSCRIBE = False
WHISPER_MODEL = "mlx-community/whisper-large-v3-turbo-asr-fp16"

# LongCat cloning knobs (per the mlx-audio LongCat README). These are the
# global defaults; any character may override cfg_strength / steps / seed in its
# CHARACTERS entry (the values below were tuned per-voice by ear).
GUIDANCE_METHOD = "apg"   # "apg" gives the best speaker/accent similarity
CFG_STRENGTH = 4.0        # higher = stronger accent adherence, but can get choppy
STEPS = 32                # ODE denoising steps: higher = smoother, slower
SEED = 1024               # diffusion seed; some seeds simply realize smoother

# Generated audio is peak-normalized to this level before writing. LongCat can
# output samples above 1.0 (raw peaks of ~1.14 observed), which would clip hard
# when written to a PCM WAV. Normalizing avoids that distortion.
OUTPUT_PEAK = 0.95

# Each character maps to one accent reference. ref_audio is a path under REF_DIR.
# ref_text is the EXACT transcript of that clip -- a mismatch (e.g. a 15s clip
# labelled with one sentence) wrecks LongCat's alignment and produces choppy,
# incoherent speech, so trim the clip and ref_text to match. Optional per-voice
# overrides: cfg_strength, steps, seed.
CHARACTERS = {
    "rosa": {
        "accent": "Mexican / Latin-American English",
        # 2-sentence (~9.2s) reference: the longer prosodic context removed the
        # inter-word choppiness a 1-sentence (~4s) clip left in.
        "ref_audio": "rosa_mexican_2sent.wav",
        "ref_text": (
            "Please call Stella. Ask her to bring these things with her from the store. "
            "Six spoons of fresh snow peas, five thick slabs of blue cheese, and maybe a "
            "snack for her brother Bob."
        ),
        "cfg_strength": 3.0,
        "steps": 64,
        "seed": 7,
    },
    "elise": {
        "accent": "French-accented English",
        # 1-sentence (~5.2s) trimmed reference; sounds clean at the defaults.
        "ref_audio": "elise_french.wav",
        "ref_text": "Please call Stella. Ask her to bring these things with her from the store.",
        "cfg_strength": 4.0,
        "steps": 32,
        "seed": 1024,
    },
    "edmund": {
        "accent": "British (Received Pronunciation)",
        "ref_audio": "edmund_british.wav",
        "ref_text": "Please call Stella. Ask her to bring these things with her from the store.",
    },
    "wade": {
        "accent": "US Southern",
        "ref_audio": "wade_southern.wav",
        "ref_text": "Please call Stella. Ask her to bring these things with her from the store.",
    },
}

# The lines to speak, tagged by character. Add as many as you like.
LINES = [
    ("rosa",   "You think I came all this way just to turn around now?"),
    ("rosa",   "Hand me the lantern. We are not waiting for morning."),
    ("elise",  "I have read your file. It does not flatter you."),
    ("edmund", "One does not simply walk into the archive uninvited."),
    ("wade",   "Well now, that's a mighty big assumption for a Tuesday."),
]

# --------------------------------------------------------------------------- #
# IMPLEMENTATION
# --------------------------------------------------------------------------- #


def load_tts():
    from mlx_audio.tts.utils import load
    print(f"Loading TTS model: {MODEL}")
    return load(MODEL)


def load_ref_audio(model, ref_path: Path):
    """Decode a reference WAV into a 24 kHz waveform array.

    LongCat's generate() expects ref_audio as the decoded samples, not a file
    path -- np.array("foo.wav") would silently become a 0-d string array. This
    mirrors what mlx_audio's own generate() wrapper does internally.
    """
    from mlx_audio.utils import load_audio
    return load_audio(str(ref_path), sample_rate=model.sample_rate)


def load_whisper():
    # Imported lazily so a normal run without AUTO_TRANSCRIBE needs nothing extra.
    from mlx_audio.stt.generate import generate_transcription
    return generate_transcription


def resolve_ref_text(character: str, cfg: dict, transcriber) -> str:
    if not AUTO_TRANSCRIBE:
        return cfg["ref_text"]
    ref_path = REF_DIR / cfg["ref_audio"]
    print(f"  transcribing reference for '{character}' ...")
    result = transcriber(model=WHISPER_MODEL, audio=str(ref_path))
    text = result.text.strip()
    print(f"  ref_text -> {text!r}")
    return text


def synth(
    model,
    text: str,
    ref_audio,
    ref_text: str,
    cfg_strength: float = CFG_STRENGTH,
    steps: int = STEPS,
    seed: int = SEED,
) -> np.ndarray:
    """One line -> waveform (float32 numpy array), peak-normalized.

    ref_audio is a pre-decoded 24 kHz waveform array (see load_ref_audio).
    """
    result = next(
        model.generate(
            text=text,
            ref_audio=ref_audio,
            ref_text=ref_text,
            guidance_method=GUIDANCE_METHOD,
            cfg_strength=cfg_strength,
            steps=steps,
            seed=seed,
        )
    )
    # result.audio is an mx.array; numpy handles the conversion for writing.
    audio = np.asarray(result.audio, dtype=np.float32)
    peak = float(np.max(np.abs(audio)))
    if peak > 0:
        audio = audio * (OUTPUT_PEAK / peak)
    return audio.astype(np.float32)


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # A character is usable only if its reference clip is present. Missing
    # clips are skipped per-character (with a warning) rather than aborting the
    # whole run, so characters whose refs you haven't sourced yet don't block
    # the ones that are ready.
    available = {
        name: cfg
        for name, cfg in CHARACTERS.items()
        if (REF_DIR / cfg["ref_audio"]).exists()
    }
    missing = [name for name in CHARACTERS if name not in available]
    if missing:
        print("Skipping characters with missing reference clips: "
              + ", ".join(missing))
        for name in missing:
            print(f"  {name}: {REF_DIR / CHARACTERS[name]['ref_audio']}")
        print()
    if not available:
        print(f"No reference clips found. Put trimmed mono WAVs in '{REF_DIR}/'.")
        return 1

    model = load_tts()
    transcriber = load_whisper() if AUTO_TRANSCRIBE else None

    # Resolve each available character's ref_text and decode its reference clip
    # once, so every line for that character reuses the same loaded waveform.
    ref_text_by_char = {
        name: resolve_ref_text(name, cfg, transcriber)
        for name, cfg in available.items()
    }
    ref_audio_by_char = {
        name: load_ref_audio(model, REF_DIR / cfg["ref_audio"])
        for name, cfg in available.items()
    }

    counters: dict[str, int] = {}
    skipped = 0
    for character, text in LINES:
        if character not in CHARACTERS:
            print(f"Skipping line for unknown character '{character}'")
            skipped += 1
            continue
        if character not in available:
            print(f"Skipping '{character}' line (no reference clip): {text!r}")
            skipped += 1
            continue
        cfg = CHARACTERS[character]
        idx = counters.get(character, 0)
        counters[character] = idx + 1

        out_path = OUTPUT_DIR / f"{character}_{idx:03d}.wav"
        print(f"[{character}] ({cfg['accent']}) -> {out_path.name}: {text!r}")
        audio = synth(
            model,
            text,
            ref_audio_by_char[character],
            ref_text_by_char[character],
            cfg_strength=cfg.get("cfg_strength", CFG_STRENGTH),
            steps=cfg.get("steps", STEPS),
            seed=cfg.get("seed", SEED),
        )
        sf.write(str(out_path), audio, SAMPLE_RATE)

    total = sum(counters.values())
    note = f" ({skipped} line(s) skipped)" if skipped else ""
    print(f"\nDone. Wrote {total} file(s) to '{OUTPUT_DIR}/'.{note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
