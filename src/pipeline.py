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
from pydub import AudioSegment

from src import config
from src.analysis import detect_questions_and_responses, compute_silence
from src.diarization import embed_segment, cluster_and_label, label_speakers
from src.metrics import compute_metrics
from src.summary import generate_summary, extractive_highlights
from src.transcription import WhisperSegment, transcribe, transcribe_stream

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


def process_audio_streaming(
    audio_path: str,
    model_size: str = config.WHISPER_MODEL_SIZE_FAST,
    language: str | None = config.WHISPER_LANGUAGE,
    max_duration_sec: float | None = None,
    teacher_name: str | None = None,
    chunk_seconds: float = config.STREAM_CHUNK_SECONDS,
):
    """Same output shape as process_audio, but yields an incremental
    result roughly every `chunk_seconds` of newly-transcribed audio
    instead of returning once at the end - so a 1-hour recording starts
    showing transcript and metrics within seconds instead of after the
    full file finishes.

    Each yielded dict is recomputed over *all* segments seen so far (not
    just the latest chunk): teacher/student role assignment naturally
    refines as more evidence comes in, and every metric is always a
    correct "running total", not a per-chunk fragment that would need
    stitching together by the caller. The last yielded dict has
    is_final=True.
    """
    wav, sr = _load_wav_mono_16k(audio_path)
    if max_duration_sec:
        wav = wav[: int(max_duration_sec * sr)]

    gen = transcribe_stream(audio_path, model_size=model_size, language=language, max_duration_sec=max_duration_sec)
    info = next(gen)
    total_duration = min(info.duration, max_duration_sec) if max_duration_sec else info.duration

    all_segments: list[WhisperSegment] = []
    all_embeddings = []
    last_flush_end = 0.0

    def build_partial(processed_duration: float, is_final: bool) -> dict:
        turns = cluster_and_label(all_segments, all_embeddings)
        analyzed = detect_questions_and_responses(turns)
        silence_sec = compute_silence(turns, processed_duration)
        metrics = compute_metrics(analyzed, processed_duration, silence_sec)
        summary_text = generate_summary(metrics, teacher_name=teacher_name)
        highlights = extractive_highlights(analyzed)
        return {
            "audio_file": os.path.basename(audio_path),
            "language": info.language,
            "language_probability": info.language_probability,
            "duration_sec": processed_duration,
            "total_duration_sec": total_duration,
            "turns": [asdict(t) for t in analyzed],
            "metrics": metrics.as_dict(),
            "summary": summary_text,
            "highlights": highlights,
            "is_final": is_final,
        }

    for seg in gen:
        all_segments.append(seg)
        all_embeddings.append(embed_segment(wav, sr, seg))
        if seg.end - last_flush_end >= chunk_seconds:
            last_flush_end = seg.end
            yield build_partial(processed_duration=seg.end, is_final=False)

    yield build_partial(processed_duration=total_duration, is_final=True)


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
