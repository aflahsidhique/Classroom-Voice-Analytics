"""Classroom-analysis heuristics on top of labeled speaker turns:
question detection, student-response matching, and silence measurement.
"""
from __future__ import annotations

from dataclasses import dataclass

from src import config
from src.diarization import Turn


@dataclass
class AnalyzedTurn:
    start: float
    end: float
    text: str
    speaker: str
    is_question: bool = False
    is_response: bool = False


def _looks_like_question(text: str) -> bool:
    t = text.strip().lower()
    if not t:
        return False
    if config.QUESTION_MARK in t:
        return True
    return any(kw in t for kw in config.QUESTION_KEYWORDS)


def detect_questions_and_responses(turns: list[Turn]) -> list[AnalyzedTurn]:
    analyzed = [
        AnalyzedTurn(t.start, t.end, t.text, t.speaker, is_question=_looks_like_question(t.text))
        for t in turns
    ]

    pending_question_end: float | None = None
    for turn in analyzed:
        if turn.speaker == "Teacher":
            if turn.is_question:
                pending_question_end = turn.end
            # a non-question teacher turn does not clear a pending question -
            # the teacher may pause and then call on a student
            continue

        # Student turn
        if pending_question_end is not None and (turn.start - pending_question_end) <= config.RESPONSE_WINDOW_SEC:
            turn.is_response = True
            pending_question_end = None  # one response credited per question

    return analyzed


def compute_silence(turns: list[Turn], total_duration: float) -> float:
    """Total seconds with no detected speech: leading/trailing gaps plus
    gaps between turns above the noise-floor threshold."""
    if not turns:
        return total_duration

    ordered = sorted(turns, key=lambda t: t.start)
    silence = ordered[0].start  # before first speech
    for prev, curr in zip(ordered, ordered[1:]):
        gap = curr.start - prev.end
        if gap >= config.MIN_SILENCE_GAP_SEC:
            silence += gap
    silence += max(0.0, total_duration - ordered[-1].end)  # after last speech
    return silence
