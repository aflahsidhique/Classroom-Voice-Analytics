"""Central configuration and tunable thresholds for the analytics pipeline.

Kept in one place so the heuristics can be explained and adjusted without
hunting through the pipeline code.
"""

# --- Transcription ---
WHISPER_MODEL_SIZE = "small"          # tiny/base/small/medium/large-v3 (CPU-friendly default)
WHISPER_MODEL_SIZE_FAST = "base"      # used for quick/live-demo mode (hosted, low-RAM)
WHISPER_COMPUTE_TYPE = "int8"         # int8 keeps CPU inference light
WHISPER_LANGUAGE = None               # None = auto-detect. Force "hi" to pin Hindi.
WHISPER_BEAM_SIZE = 5
VAD_FILTER = True                     # drop non-speech before transcribing

# --- Streaming ---
# process_audio_streaming() flushes an incremental result after roughly
# this many new seconds of transcribed audio, so a caller (UI, CLI) can
# show progress instead of waiting for the whole file.
STREAM_CHUNK_SECONDS = 45.0

# --- Speaker turn construction ---
# Whisper segments are merged into a "turn" as long as the same speaker
# keeps talking and the gap between segments is below this threshold.
TURN_MERGE_GAP_SEC = 1.5

# --- Diarization (speaker clustering) ---
# We cluster segment-level voice embeddings into this many raw clusters,
# then collapse every cluster except the single largest-total-duration one
# into "Student". This approximates Teacher-vs-Student without needing to
# know the true speaker count up front (classrooms have 1 teacher, many
# students speaking briefly/one after another).
MAX_RAW_SPEAKER_CLUSTERS = 4
MIN_SEGMENT_DURATION_FOR_EMBEDDING = 0.4  # seconds; shorter clips are too noisy to embed reliably

# --- Question / response heuristics ---
QUESTION_MARK = "?"
QUESTION_KEYWORDS = [
    # English
    "what", "why", "how", "when", "where", "who", "which", "whose",
    "can you", "could you", "do you", "does anyone", "is it", "are you",
    "will you", "should", "right?", "okay?", "understand?", "clear?",
    "anyone know", "tell me", "explain",
    # Hindi (Devanagari + common Romanized forms used in code-switched speech)
    "kya", "क्या", "kyun", "kyu", "क्यों", "kaise", "कैसे", "kab", "कब",
    "kahan", "कहाँ", "कहां", "kaun", "कौन", "kitna", "kitne", "कितना", "कितने",
    "batao", "बताओ", "samjhe", "समझे", "samajh", "समझ", "theek hai",
    "ठीक है", "hai na", "है ना",
]
# Max seconds after a teacher question ends for a following student
# turn to be credited as a "response" to that question.
RESPONSE_WINDOW_SEC = 12.0

# --- Silence ---
# Any gap between consecutive detected speech segments longer than this
# counts toward total silence duration.
MIN_SILENCE_GAP_SEC = 0.5

SAMPLE_RATE = 16000
