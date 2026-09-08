"""Audio -> text transcription using faster-whisper.

faster-whisper (CTranslate2 backend) is chosen over vanilla openai-whisper
because it runs several times faster on CPU with int8 quantization and needs
no GPU - important for an offline-first classroom tool that may run on a
school laptop rather than a cloud GPU box. Whisper is multilingual out of
the box, so Hindi (and other Indian languages) and English code-switching
within the same recording are both handled without extra training.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

from faster_whisper import WhisperModel

from src import config


@dataclass
class WhisperSegment:
    start: float
    end: float
    text: str


@dataclass
class TranscriptionResult:
    language: str
    language_probability: float
    duration: float
    segments: list[WhisperSegment] = field(default_factory=list)


@lru_cache(maxsize=2)
def _load_model(model_size: str) -> WhisperModel:
    # cached so Streamlit reruns / repeated CLI calls don't reload the model
    return WhisperModel(model_size, device="cpu", compute_type=config.WHISPER_COMPUTE_TYPE)


def transcribe(
    audio_path: str,
    model_size: str = config.WHISPER_MODEL_SIZE,
    language: str | None = config.WHISPER_LANGUAGE,
    max_duration_sec: float | None = None,
) -> TranscriptionResult:
    """Transcribe an audio file to timestamped segments.

    max_duration_sec: if set, only the first N seconds are processed. Used
    by the hosted demo to keep inference time bounded on free-tier compute.
    """
    model = _load_model(model_size)

    clip_kwargs = {}
    if max_duration_sec:
        clip_kwargs["clip_timestamps"] = f"0,{max_duration_sec}"

    segments_iter, info = model.transcribe(
        audio_path,
        language=language,
        beam_size=config.WHISPER_BEAM_SIZE,
        vad_filter=config.VAD_FILTER,
        vad_parameters={"min_silence_duration_ms": 500},
        **clip_kwargs,
    )

    segments = [
        WhisperSegment(start=s.start, end=s.end, text=s.text.strip())
        for s in segments_iter
        if s.text and s.text.strip()
    ]

    duration = min(info.duration, max_duration_sec) if max_duration_sec else info.duration
    return TranscriptionResult(
        language=info.language,
        language_probability=info.language_probability,
        duration=duration,
        segments=segments,
    )
