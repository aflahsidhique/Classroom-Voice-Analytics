"""Classroom engagement metrics.

Every metric below documents its formula, a short explanation, and how to
read the resulting value, per the assignment's requirement.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

from src.analysis import AnalyzedTurn


@dataclass
class Metric:
    name: str
    value: float
    unit: str
    formula: str
    explanation: str
    interpretation: str


@dataclass
class ClassroomMetrics:
    duration_sec: float
    teacher_talk_sec: float
    student_talk_sec: float
    silence_sec: float
    teacher_turns: int
    student_turns: int
    teacher_questions: int
    student_responses: int
    metrics: list[Metric]

    def as_dict(self) -> dict:
        d = asdict(self)
        return d


def _interpret_dominance(ratio: float) -> str:
    if ratio >= 0.70:
        return "Lecture-dominated: the teacher held the floor for most of the session."
    if ratio >= 0.45:
        return "Balanced: teacher and student talk time are reasonably even."
    return "Student-led: students spoke more than the teacher, suggesting discussion or group work."


def _interpret_participation(rate: float) -> str:
    if rate >= 0.6:
        return "High participation: most teacher questions received a student response."
    if rate >= 0.3:
        return "Moderate participation: roughly one in three questions went answered."
    return "Low participation: most teacher questions were not visibly answered - could indicate rhetorical questioning, silence, or a diarization miss."


def _interpret_interaction(density: float) -> str:
    if density >= 6:
        return "Highly interactive: frequent back-and-forth between teacher and students."
    if density >= 2:
        return "Moderately interactive: periodic exchanges within an otherwise teacher-led flow."
    return "Low interactivity: long uninterrupted stretches, closer to a monologue/lecture style."


def _interpret_silence(ratio: float) -> str:
    if ratio >= 0.25:
        return "Substantial silence: could reflect independent work, writing time, or transcription-VAD dropping quiet speech."
    if ratio >= 0.1:
        return "Normal amount of pause/silence for a live classroom recording."
    return "Very little silence: near-continuous talking throughout the session."


def compute_metrics(turns: list[AnalyzedTurn], duration_sec: float, silence_sec: float) -> ClassroomMetrics:
    teacher_turns = [t for t in turns if t.speaker == "Teacher"]
    student_turns = [t for t in turns if t.speaker == "Student"]

    teacher_talk = sum(t.end - t.start for t in teacher_turns)
    student_talk = sum(t.end - t.start for t in student_turns)
    total_talk = teacher_talk + student_talk

    teacher_questions = sum(1 for t in teacher_turns if t.is_question)
    student_responses = sum(1 for t in student_turns if t.is_response)

    # interaction count = number of times speaker role switches (teacher->student or student->teacher)
    interaction_count = sum(
        1 for prev, curr in zip(turns, turns[1:]) if prev.speaker != curr.speaker
    )
    duration_min = duration_sec / 60.0 if duration_sec else 1.0

    dominance_ratio = (teacher_talk / total_talk) if total_talk > 0 else 0.0
    participation_rate = (student_responses / teacher_questions) if teacher_questions > 0 else 0.0
    interaction_density = interaction_count / duration_min if duration_min > 0 else 0.0
    silence_ratio = (silence_sec / duration_sec) if duration_sec > 0 else 0.0

    metrics = [
        Metric(
            name="Teacher Dominance Ratio",
            value=round(dominance_ratio, 3),
            unit="ratio (0-1)",
            formula="teacher_talk_time / (teacher_talk_time + student_talk_time)",
            explanation="Share of total classroom speech that came from the teacher.",
            interpretation=_interpret_dominance(dominance_ratio),
        ),
        Metric(
            name="Student Participation Rate",
            value=round(participation_rate, 3),
            unit="ratio (0-1, can exceed 1)",
            formula="student_responses / teacher_questions",
            explanation="Fraction of teacher questions that were followed by a student turn within "
                        f"{12:.0f}s (a proxy for an answer).",
            interpretation=_interpret_participation(participation_rate),
        ),
        Metric(
            name="Interaction Density",
            value=round(interaction_density, 2),
            unit="speaker switches / minute",
            formula="count(speaker role changes) / duration_minutes",
            explanation="How often the floor alternates between teacher and student(s) per minute.",
            interpretation=_interpret_interaction(interaction_density),
        ),
        Metric(
            name="Silence Ratio",
            value=round(silence_ratio, 3),
            unit="ratio (0-1)",
            formula="silence_time / total_duration",
            explanation="Portion of the recording with no detected speech from anyone.",
            interpretation=_interpret_silence(silence_ratio),
        ),
    ]

    return ClassroomMetrics(
        duration_sec=duration_sec,
        teacher_talk_sec=teacher_talk,
        student_talk_sec=student_talk,
        silence_sec=silence_sec,
        teacher_turns=len(teacher_turns),
        student_turns=len(student_turns),
        teacher_questions=teacher_questions,
        student_responses=student_responses,
        metrics=metrics,
    )
