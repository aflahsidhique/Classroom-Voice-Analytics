"""CLI entry point: process one classroom audio file and cache the result.

Usage:
    python scripts/run_pipeline.py path/to/audio.mp3 --name classroom_01
    python scripts/run_pipeline.py path/to/audio.mp3 --model tiny --max-duration 300
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.pipeline import process_and_cache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio_path", help="Path to the classroom audio file (mp3/wav/...)")
    parser.add_argument("--name", default=None, help="Cache name for outputs/<name>.json (default: filename)")
    parser.add_argument("--model", default=config.WHISPER_MODEL_SIZE, help="Whisper model size")
    parser.add_argument("--language", default=config.WHISPER_LANGUAGE, help="Force language code, e.g. hi (default: auto-detect)")
    parser.add_argument("--max-duration", type=float, default=None, help="Only process the first N seconds")
    parser.add_argument("--teacher-name", default=None, help="Optional teacher name/label for the summary text")
    args = parser.parse_args()

    print(f"Processing {args.audio_path} with whisper '{args.model}' ...")
    result = process_and_cache(
        args.audio_path,
        cache_name=args.name,
        model_size=args.model,
        language=args.language,
        max_duration_sec=args.max_duration,
        teacher_name=args.teacher_name,
    )
    print(f"Done. duration={result['duration_sec']:.1f}s language={result['language']}")
    print(f"Cached to outputs/{args.name or Path(args.audio_path).stem}.json")
    print()
    print(result["summary"])


if __name__ == "__main__":
    main()
