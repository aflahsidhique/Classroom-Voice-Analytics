"""Human-readable classroom summary generation.

Deliberately template-based rather than calling an LLM API: it keeps the
whole pipeline offline-capable (no API key, no per-request cost, works on a
school laptop with no internet) and produces a fully deterministic summary
straight from the numbers we already trust, instead of asking a generative
model to "describe" statistics it could get wrong.
"""
from __future__ import annotations

from src.metrics import ClassroomMetrics


def _fmt_minutes(seconds: float) -> str:
    return f"{seconds / 60:.1f} min"


def generate_summary(metrics: ClassroomMetrics, teacher_name: str | None = None) -> str:
    teacher_label = teacher_name or "The teacher"
    dominance = next(m for m in metrics.metrics if m.name == "Teacher Dominance Ratio")
    participation = next(m for m in metrics.metrics if m.name == "Student Participation Rate")
    interaction = next(m for m in metrics.metrics if m.name == "Interaction Density")
    silence = next(m for m in metrics.metrics if m.name == "Silence Ratio")

    lines = [
        f"This session ran for {_fmt_minutes(metrics.duration_sec)}. "
        f"{teacher_label} spoke for {_fmt_minutes(metrics.teacher_talk_sec)} "
        f"({dominance.value:.0%} of total talk time) across {metrics.teacher_turns} turns; "
        f"students spoke for {_fmt_minutes(metrics.student_talk_sec)} across {metrics.student_turns} turns.",

        f"{teacher_label} asked an estimated {metrics.teacher_questions} question(s), "
        f"of which {metrics.student_responses} were followed by a student response "
        f"({participation.value:.0%} response rate).",

        f"The floor switched between teacher and students about {interaction.value:.1f} times per minute "
        f"({interaction.interpretation.split(':')[0].lower()}).",

        f"Roughly {silence.value:.0%} of the recording had no detected speech "
        f"({_fmt_minutes(metrics.silence_sec)} of silence/noise).",

        dominance.interpretation,
    ]
    return " ".join(lines)


def extractive_highlights(turns, top_n: int = 5) -> list[str]:
    """Very lightweight extractive highlight picker: longest teacher
    questions and longest student turns, as a cheap stand-in for a content
    summary without pulling in a full NLP summarization model."""
    questions = sorted(
        (t for t in turns if t.speaker == "Teacher" and t.is_question),
        key=lambda t: len(t.text),
        reverse=True,
    )[:top_n]
    student_answers = sorted(
        (t for t in turns if t.speaker == "Student" and t.is_response),
        key=lambda t: len(t.text),
        reverse=True,
    )[:top_n]
    highlights = []
    for q in questions:
        highlights.append(f"Q ({q.start:.0f}s): {q.text}")
    for a in student_answers:
        highlights.append(f"A ({a.start:.0f}s): {a.text}")
    return highlights
