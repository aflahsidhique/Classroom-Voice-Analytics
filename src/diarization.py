"""Approximate Teacher-vs-Student speaker labeling.

Full speaker diarization (pyannote-style) needs a gated HuggingFace model
and a HF auth token, which breaks the "clone and run" offline-first goal of
this prototype. Instead we do a lighter, dependency-free-of-tokens version
that is good enough for classroom analytics, where we only need a *role*
(Teacher / Student) rather than each individual student's identity:

1. Take the timestamped Whisper segments.
2. Compute a voice-embedding (Resemblyzer d-vector) for each segment's
   audio slice.
3. Cluster the embeddings into a handful of raw speaker clusters.
4. The single cluster with the greatest total speaking duration is almost
   always the teacher (one person, dominates classroom talk time) and is
   labeled "Teacher"; every other cluster is collapsed into "Student".

This matches the assignment's explicit allowance for approximation on the
Teacher vs Student split.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from resemblyzer import VoiceEncoder, preprocess_wav
from sklearn.cluster import AgglomerativeClustering

from src import config
from src.transcription import WhisperSegment

_encoder: VoiceEncoder | None = None


def _get_encoder() -> VoiceEncoder:
    global _encoder
    if _encoder is None:
        _encoder = VoiceEncoder()
    return _encoder


@dataclass
class Turn:
    start: float
    end: float
    text: str
    speaker: str  # "Teacher" | "Student"

    @property
    def duration(self) -> float:
        return self.end - self.start


def _embed_segments(wav: np.ndarray, sr: int, segments: list[WhisperSegment]) -> np.ndarray:
    encoder = _get_encoder()
    embeddings = []
    for seg in segments:
        start_i, end_i = int(seg.start * sr), int(seg.end * sr)
        clip = wav[start_i:end_i]
        if len(clip) < sr * config.MIN_SEGMENT_DURATION_FOR_EMBEDDING:
            embeddings.append(None)
            continue
        processed = preprocess_wav(clip, source_sr=sr)
        if len(processed) == 0:
            embeddings.append(None)
            continue
        embeddings.append(encoder.embed_utterance(processed))
    return embeddings


def label_speakers(wav: np.ndarray, sr: int, segments: list[WhisperSegment]) -> list[Turn]:
    """Assign a Teacher/Student role to every Whisper segment."""
    if not segments:
        return []

    embeddings = _embed_segments(wav, sr, segments)
    valid_idx = [i for i, e in enumerate(embeddings) if e is not None]

    if len(valid_idx) < 2:
        # Not enough signal to cluster - fall back to "everyone is the
        # dominant speaker" so the pipeline still produces output.
        raw_labels = [0] * len(segments)
    else:
        n_clusters = min(config.MAX_RAW_SPEAKER_CLUSTERS, len(valid_idx))
        X = np.stack([embeddings[i] for i in valid_idx])
        clustering = AgglomerativeClustering(n_clusters=n_clusters, metric="cosine", linkage="average")
        cluster_ids = clustering.fit_predict(X)

        raw_labels = [-1] * len(segments)
        for idx, cid in zip(valid_idx, cluster_ids):
            raw_labels[idx] = int(cid)
        # segments with no embedding inherit the previous segment's cluster
        last = 0
        for i in range(len(raw_labels)):
            if raw_labels[i] == -1:
                raw_labels[i] = last
            else:
                last = raw_labels[i]

    # Total duration per raw cluster -> largest cluster becomes "Teacher"
    durations: dict[int, float] = {}
    for seg, cid in zip(segments, raw_labels):
        durations[cid] = durations.get(cid, 0.0) + (seg.end - seg.start)
    teacher_cluster = max(durations, key=durations.get)

    turns = [
        Turn(
            start=seg.start,
            end=seg.end,
            text=seg.text,
            speaker="Teacher" if cid == teacher_cluster else "Student",
        )
        for seg, cid in zip(segments, raw_labels)
    ]
    return _merge_adjacent_turns(turns)


def _merge_adjacent_turns(turns: list[Turn]) -> list[Turn]:
    """Merge consecutive same-speaker segments separated by a short gap
    into a single turn, so the transcript reads naturally and turn-taking
    counts reflect real exchanges rather than Whisper's raw chunking."""
    if not turns:
        return []
    merged = [turns[0]]
    for t in turns[1:]:
        prev = merged[-1]
        gap = t.start - prev.end
        if t.speaker == prev.speaker and gap <= config.TURN_MERGE_GAP_SEC:
            merged[-1] = Turn(
                start=prev.start,
                end=t.end,
                text=(prev.text + " " + t.text).strip(),
                speaker=prev.speaker,
            )
        else:
            merged.append(t)
    return merged
