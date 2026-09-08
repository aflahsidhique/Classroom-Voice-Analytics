import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis import AnalyzedTurn, detect_questions_and_responses, compute_silence
from src.diarization import Turn
from src.metrics import compute_metrics


def test_question_detection_marks_question_mark_and_keywords():
    turns = [
        Turn(0, 2, "What is photosynthesis?", "Teacher"),
        Turn(2, 4, "kya aap samajh gaye", "Teacher"),
        Turn(4, 6, "The sun is bright", "Teacher"),
    ]
    analyzed = detect_questions_and_responses(turns)
    assert analyzed[0].is_question is True
    assert analyzed[1].is_question is True
    assert analyzed[2].is_question is False


def test_response_credited_within_window():
    turns = [
        Turn(0, 2, "What is 2 plus 2?", "Teacher"),
        Turn(3, 4, "Four", "Student"),
    ]
    analyzed = detect_questions_and_responses(turns)
    assert analyzed[1].is_response is True


def test_response_not_credited_outside_window():
    turns = [
        Turn(0, 2, "What is 2 plus 2?", "Teacher"),
        Turn(60, 61, "Four", "Student"),
    ]
    analyzed = detect_questions_and_responses(turns)
    assert analyzed[1].is_response is False


def test_silence_computation():
    turns = [Turn(0, 10, "hello", "Teacher"), Turn(15, 20, "hi", "Student")]
    silence = compute_silence(turns, total_duration=25)
    # gap 10-15 (5s) + tail 20-25 (5s) = 10s
    assert silence == 10


def test_teacher_dominance_ratio():
    turns = [
        AnalyzedTurn(0, 8, "lecture", "Teacher"),
        AnalyzedTurn(8, 10, "ok", "Student"),
    ]
    m = compute_metrics(turns, duration_sec=10, silence_sec=0)
    dominance = next(x for x in m.metrics if x.name == "Teacher Dominance Ratio")
    assert dominance.value == 0.8


def test_no_teacher_questions_gives_zero_participation():
    turns = [AnalyzedTurn(0, 5, "statement", "Teacher")]
    m = compute_metrics(turns, duration_sec=5, silence_sec=0)
    participation = next(x for x in m.metrics if x.name == "Student Participation Rate")
    assert participation.value == 0
