"""CLI entry point: process one classroom audio file and cache the result.

Usage:
    python scripts/run_pipeline.py path/to/audio.mp3 --name classroom_01
    python scripts/run_pipeline.py path/to/audio.mp3 --model tiny --max-duration 300
    python scripts/run_pipeline.py path/to/audio.mp3 --stream   # print progress instead of waiting silently
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.pipeline import OUTPUTS_DIR, process_and_cache, process_audio_streaming


def run_streaming(args, cache_name: str) -> dict:
    result = None
    for partial in process_audio_streaming(
        args.audio_path,
        model_size=args.model,
        language=args.language,
        max_duration_sec=args.max_duration,
        teacher_name=args.teacher_name,
    ):
        total = partial.get("total_duration_sec") or partial["duration_sec"] or 1.0
        pct = 100 * min(partial["duration_sec"] / total, 1.0)
        m = partial["metrics"]
        print(
            f"  [{pct:5.1f}%] {partial['duration_sec']:6.0f}s/{total:.0f}s processed | "
            f"teacher dominance={next(x['value'] for x in m['metrics'] if x['name'] == 'Teacher Dominance Ratio'):.2f} | "
            f"questions={m['teacher_questions']} responses={m['student_responses']}"
        )
        result = partial

    OUTPUTS_DIR.mkdir(exist_ok=True)
    cache_path = OUTPUTS_DIR / f"{cache_name}.json"
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio_path", help="Path to the classroom audio file (mp3/wav/...)")
    parser.add_argument("--name", default=None, help="Cache name for outputs/<name>.json (default: filename)")
    parser.add_argument("--model", default=config.WHISPER_MODEL_SIZE, help="Whisper model size")
    parser.add_argument("--language", default=config.WHISPER_LANGUAGE, help="Force language code, e.g. hi (default: auto-detect)")
    parser.add_argument("--max-duration", type=float, default=None, help="Only process the first N seconds")
    parser.add_argument("--teacher-name", default=None, help="Optional teacher name/label for the summary text")
    parser.add_argument(
        "--stream", action="store_true",
        help="Print progress as the audio is transcribed instead of waiting silently for the whole file",
    )
    args = parser.parse_args()
    cache_name = args.name or Path(args.audio_path).stem

    print(f"Processing {args.audio_path} with whisper '{args.model}' ...")
    if args.stream:
        result = run_streaming(args, cache_name)
    else:
        result = process_and_cache(
            args.audio_path,
            cache_name=args.name,
            model_size=args.model,
            language=args.language,
            max_duration_sec=args.max_duration,
            teacher_name=args.teacher_name,
        )
    print(f"Done. duration={result['duration_sec']:.1f}s language={result['language']}")
    print(f"Cached to outputs/{cache_name}.json")
    print()
    print(result["summary"])


if __name__ == "__main__":
    main()
