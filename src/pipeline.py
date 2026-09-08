"""End-to-end orchestration: audio file -> transcript + metrics + summary.

Results are cached to JSON on disk (outputs/<name>.json) so the demo app
never has to re-run heavy inference for a file it has already processed.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np
import soundfile as sf
from pydub import AudioSegment

from src import config
from src.analysis import detect_questions_and_responses, compute_silence
from src.diarization import label_speakers
from src.metrics import compute_metrics
from src.summary import generate_summary, extractive_highlights
from src.transcription import transcribe

OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "outputs"


def _load_wav_mono_16k(audio_path: str) -> tuple[np.ndarray, int]:
    """Decode any ffmpeg-supported audio (mp3, m4a, wav...) to mono 16kHz
    float32 PCM for embedding extraction."""
    audio = AudioSegment.from_file(audio_path)
    audio = audio.set_channels(1).set_frame_rate(config.SAMPLE_RATE)
    samples = np.array(audio.get_array_of_samples()).astype(np.float32) / 32768.0
    return samples, config.SAMPLE_RATE


def process_audio(
    audio_path: str,
    model_size: str = config.WHISPER_MODEL_SIZE,
    language: str | None = config.WHISPER_LANGUAGE,
    max_duration_sec: float | None = None,
    teacher_name: str | None = None,
) -> dict:
    transcription = transcribe(
        audio_path, model_size=model_size, language=language, max_duration_sec=max_duration_sec
    )

    wav, sr = _load_wav_mono_16k(audio_path)
    if max_duration_sec:
        wav = wav[: int(max_duration_sec * sr)]

    turns = label_speakers(wav, sr, transcription.segments)
    analyzed = detect_questions_and_responses(turns)
    silence_sec = compute_silence(turns, transcription.duration)
    metrics = compute_metrics(analyzed, transcription.duration, silence_sec)
    summary_text = generate_summary(metrics, teacher_name=teacher_name)
    highlights = extractive_highlights(analyzed)

    return {
        "audio_file": os.path.basename(audio_path),
        "language": transcription.language,
        "language_probability": transcription.language_probability,
        "duration_sec": transcription.duration,
        "turns": [asdict(t) for t in analyzed],
        "metrics": metrics.as_dict(),
        "summary": summary_text,
        "highlights": highlights,
    }


def process_and_cache(
    audio_path: str,
    cache_name: str | None = None,
    **kwargs,
) -> dict:
    OUTPUTS_DIR.mkdir(exist_ok=True)
    cache_name = cache_name or Path(audio_path).stem
    cache_path = OUTPUTS_DIR / f"{cache_name}.json"

    result = process_audio(audio_path, **kwargs)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result


def load_cached(cache_name: str) -> dict | None:
    cache_path = OUTPUTS_DIR / f"{cache_name}.json"
    if not cache_path.exists():
        return None
    with open(cache_path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_cached() -> list[str]:
    OUTPUTS_DIR.mkdir(exist_ok=True)
    return sorted(p.stem for p in OUTPUTS_DIR.glob("*.json"))
