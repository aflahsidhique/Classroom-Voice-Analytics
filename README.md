---
title: Classroom Voice Analytics
emoji: 🎙️
colorFrom: blue
colorTo: yellow
sdk: streamlit
sdk_version: "1.38.0"
app_file: app.py
pinned: false
---

# Classroom Voice Analytics MVP

Converts raw classroom audio into a speaker-labeled transcript, teacher/student
engagement metrics, and an auto-generated classroom summary — built as the
MakerGhat Full-Stack Developer pre-work assignment.

**Live demo:** _[add Streamlit Cloud URL after deployment — see §8]_
**Repo:** https://github.com/aflahsidhique/Classroom-Voice-Analytics

---

## 1. What it does

1. **Transcription** — `faster-whisper` (CTranslate2 Whisper) turns the audio into
   timestamped text segments. Whisper is multilingual, so Hindi and
   English-Hindi code-switching (very common in Indian classrooms) are handled
   without any extra training; any other Whisper-supported Indian language works
   by passing `--language`.
2. **Teacher/Student labeling** — each segment gets a voice embedding
   (`resemblyzer`), the embeddings are clustered, and the cluster with the most
   total speaking time is labeled `Teacher`, everything else `Student`. This
   avoids needing a gated diarization model or an auth token, at the cost of not
   telling individual students apart — which the assignment explicitly allows
   ("approximation is acceptable").
3. **Classroom analysis** — keyword + punctuation heuristics flag teacher
   questions; a student turn that starts within 12s of a question is credited as
   a response; silence is whatever isn't covered by a detected speech segment.
4. **Engagement metrics** — see [§3](#3-engagement-metrics).
5. **Summary** — a deterministic, template-based paragraph generated straight
   from the computed stats (no LLM call — offline, free, reproducible), plus a
   handful of extracted highlight lines (longest questions / longest answers).
6. **Demo UI** — two independent frontends on the same pipeline: a Streamlit
   app (`app.py`) and a plain HTML/CSS/JS + Flask app (`webapp/`) — see
   [§2b](#2b-two-frontends).
7. **Streaming** — uploaded audio is processed incrementally (see [§2a](#2a-streaming-design)),
   so a long recording starts showing transcript and live-updating metrics
   within seconds instead of only after the whole file finishes.

## 2a. Streaming design

`faster-whisper` already decodes and yields segments lazily as it works through
the audio — it doesn't wait for the whole file before returning anything.
`process_audio_streaming()` (`src/pipeline.py`) consumes that generator
directly and, roughly every 45 seconds of newly-transcribed audio
(`STREAM_CHUNK_SECONDS` in `config.py`), re-runs speaker clustering and every
metric over *everything seen so far* and yields a fresh partial result.

This was a deliberate choice over the more obvious-looking alternative of
manually slicing the audio file into separate ~45s chunks and calling Whisper
on each one independently: pre-slicing loses cross-chunk context (worse
transcription right at each cut, and a sentence spanning a chunk boundary gets
mangled), needs N separate model invocations instead of one, and duplicates
work Whisper is already doing internally. Piggybacking on Whisper's own
lazy generator gets the identical "results within seconds" UX for free, with
none of that downside.

On the UI side (`app.py`), there's no custom WebSocket server — Streamlit's
existing connection between server and browser already pushes incremental
updates as `st.empty()` placeholders are rewritten during a single script
run, so re-rendering the transcript/metrics inside the `for partial in
process_audio_streaming(...)` loop is enough to get the same live-updating
effect without a second piece of server infrastructure. The CLI
(`scripts/run_pipeline.py --stream`) uses the same generator to print
progress instead of running silently.

One consequence worth knowing: because every yielded result re-clusters
*all* segments so far (not just the newest chunk), the Teacher/Student role
assignment and the metrics can shift slightly between early updates as more
evidence about who talks the most arrives — e.g. dominance flipping from
"student-led" to "lecture-dominated" a few chunks in. This is expected: each
individual update is internally consistent (a correct running total, not a
fragment), it just isn't monotonic while there's minimal data. It naturally
settles down once a few chunks have accumulated.

## 2b. Two frontends

Both frontends are thin UI layers over the same `src/` pipeline - neither
contains any transcription/diarization/metrics logic itself:

| | `app.py` (Streamlit) | `webapp/` (Flask) |
|---|---|---|
| Push mechanism | Streamlit's own server↔browser connection, driven by rewriting `st.empty()` placeholders in a loop | Server-Sent Events (`text/event-stream`) consumed by the browser's native `EventSource` |
| Why this and not the other option | Zero frontend code to write - Python-only | Full control over markup/styling; a plain HTML page has no framework runtime tying it to one host |
| Best for | Fastest way to get a working demo | A more typical full-stack shape (REST-ish upload endpoint + a real HTML/CSS/JS client) |

Why SSE and not a raw WebSocket for the Flask app: updates only ever flow
server → browser (upload is a plain POST, everything after is push), which
is exactly what SSE is for. A WebSocket would add Flask-SocketIO plus an
async worker (eventlet/gevent) for two-way messaging this app never uses;
SSE works with Flask's ordinary WSGI request/response model and needs
nothing beyond the standard library's `queue` for the producer/consumer
hand-off between the background transcription thread and the streaming
response.

**Running it:**
```bash
python webapp/flask_app.py   # http://localhost:5000
```

**Important constraint if you ever deploy this behind a real WSGI server**
(gunicorn, waitress, ...): job state (`_jobs` in `flask_app.py`) is a plain
in-process dict, not shared across processes. This bit us once already
during development - `debug=True`'s default reloader spawns a *second*
Python process on file changes (via `subprocess`, since Windows can't
`fork`), and that second process had its own empty `_jobs` dict, so a job
created by the first process was invisible to the second and every upload
failed with "unknown job". The fix here was `use_reloader=False`; the same
class of bug will reappear with `gunicorn -w N` for any `N > 1`, so run it
with a **single worker** (`gunicorn -w 1 --threads 8 -b 0.0.0.0:8000
webapp.flask_app:app` - `--threads` still lets multiple SSE connections and
uploads be served concurrently within that one process). Moving the job
queue to something external (Redis, etc.) would remove this constraint, but
is out of scope for this MVP.

## 2. Architecture

```
audio file (mp3/wav/...)
      │
      ▼
┌─────────────────┐   faster-whisper (CPU, int8)
│  transcription   │   -> timestamped text segments
└─────────────────┘
      │
      ▼
┌─────────────────┐   resemblyzer embeddings + clustering
│   diarization    │   -> Teacher/Student per segment, merged into turns
└─────────────────┘
      │
      ▼
┌─────────────────┐   keyword/regex + timing window
│    analysis      │   -> is_question, is_response, silence
└─────────────────┘
      │
      ▼
┌─────────────────┐
│     metrics      │   -> 4 engagement metrics with formula + interpretation
└─────────────────┘
      │
      ▼
┌─────────────────┐
│     summary       │  -> template-based classroom summary
└─────────────────┘
      │
      ▼
outputs/<name>.json  ──▶  app.py (Streamlit)  or  webapp/flask_app.py (Flask + SSE)
```

Code layout:

```
src/
  config.py         tunable thresholds (all in one place, documented)
  transcription.py  faster-whisper wrapper (blocking + streaming generator)
  diarization.py    voice-embedding clustering -> Teacher/Student turns
  analysis.py       question + response detection, silence
  metrics.py        engagement metric formulas/interpretation
  summary.py        stats-based summary + highlight extraction
  pipeline.py        orchestrates the above, caches JSON to outputs/
scripts/
  run_pipeline.py             CLI: process one audio file (add --stream for progress)
  generate_sample_audio.ps1   builds the synthetic demo clip (see §5)
app.py               Streamlit demo interface
webapp/
  flask_app.py        Flask + Server-Sent Events demo interface (see §2b)
  templates/index.html, static/{style.css,app.js}
tests/               unit tests for the analysis/metrics heuristics
```

## 3. Engagement metrics

| Metric | Formula | What it means | How to read it |
|---|---|---|---|
| **Teacher Dominance Ratio** | `teacher_talk_time / (teacher_talk_time + student_talk_time)` | Share of speech that came from the teacher. | ≥0.70 lecture-dominated · 0.45–0.70 balanced · <0.45 student-led |
| **Student Participation Rate** | `student_responses / teacher_questions` | Fraction of teacher questions followed by a student turn within 12s. | ≥0.6 high · 0.3–0.6 moderate · <0.3 low |
| **Interaction Density** | `count(speaker role switches) / duration_minutes` | How often the floor alternates between teacher and students, per minute. | ≥6 highly interactive · 2–6 moderate · <2 monologue-like |
| **Silence Ratio** *(optional metric)* | `silence_time / total_duration` | Portion of the recording with no detected speech. | ≥0.25 substantial · 0.10–0.25 normal · <0.10 near-continuous talking |

All four are computed in [`src/metrics.py`](src/metrics.py); interpretation
thresholds are heuristic starting points meant to be tuned against real
classroom baselines, not fixed pedagogical truths.

## 4. Design decisions & trade-offs

- **faster-whisper over pyannote for speaker roles.** Full diarization
  (pyannote) needs a gated HuggingFace model + auth token, which breaks
  "clone and run" and doesn't fit an offline-first product story for schools
  with unreliable internet. Since the assignment only needs a *role*
  (Teacher/Student), not individual student identity, clustering +
  largest-cluster-is-teacher is a much lighter, token-free approximation.
- **Deterministic summary instead of an LLM call.** The "Classroom Summary" is
  generated from the metrics we already computed and trust, not by asking a
  generative model to describe numbers (which risks hallucinated figures). It
  also means the whole pipeline needs zero API keys and zero internet at
  inference time.
- **Two-tier compute for the hosted demo.** A 1-hour classroom recording with
  the `small`/`medium` Whisper model is too slow for free-tier hosting (Streamlit
  Community Cloud gives ~1 CPU core, 1GB RAM). So the live demo:
  - defaults to **precomputed** results for sample sessions (instant), and
  - for user uploads, runs the `base` model on **at most the first 4 minutes**
    of audio.
  Full-length, higher-accuracy processing is one CLI command locally
  (`python scripts/run_pipeline.py your_audio.mp3`), where model size and
  duration limits are yours to set in `src/config.py`.
- **Turn merging.** Whisper naturally breaks speech into short segments; we
  don't want a single sentence to be counted as five separate "turns", so
  consecutive same-speaker segments closer together than 1.5s are merged
  before any counting happens (`TURN_MERGE_GAP_SEC` in `config.py`).

## 5. About the sample audio & data privacy

The assignment's Google Drive folder contains **real classroom recordings**
with children's voices, and each session's metadata includes the recording
teacher's name and the **exact GPS coordinates of the school**. Those files
were used locally to build and sanity-check the pipeline (including
verifying real-world transcription behavior), but they are **not committed to
this repository and not used in the public live demo** — publishing a minor's
voice recording plus a precise school location publicly isn't something this
project does without explicit consent from the school/guardians.

Instead, `data/sample/classroom_sample.mp3` is a **synthetic** ~80-second
teacher/student dialogue generated locally with Windows' built-in
text-to-speech (`scripts/generate_sample_audio.ps1`, two distinct voices) —
safe to publish and to run through the full pipeline in the live demo. The
real recordings can be dropped into `data/raw_audio/` (already gitignored)
and processed with `scripts/run_pipeline.py` for local-only evaluation.

## 6. Running it locally

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt

# Build the bundled synthetic sample (~80s) and process it:
python scripts/run_pipeline.py data/sample/classroom_sample.mp3 --name classroom_sample

# Process your own audio (auto-detects language; force with --language hi):
python scripts/run_pipeline.py path/to/audio.mp3 --name my_session

# Same, but print progress instead of waiting silently for the whole file:
python scripts/run_pipeline.py path/to/audio.mp3 --name my_session --stream

# Launch the demo UI:
streamlit run app.py
```

Processing time on CPU is roughly real-time-ish for the `small` model
(~1 hour audio ≈ 30-60 min transcription depending on hardware); use
`--model tiny` or `--max-duration 300` for a quick smoke test, or `--stream`
to watch progress rather than waiting on a silent terminal either way.

## 7. Testing

```bash
pytest tests/ -q
```

Unit tests cover the question/response detection heuristics, silence
computation, and the metric formulas — the parts most likely to have an
off-by-one or wrong-threshold bug, and the parts you can verify without
running the ML models at all.

## 8. Deployment

**Option A — Streamlit Community Cloud (recommended, easiest):**

1. Push this repo to GitHub (public, since Streamlit Cloud's free tier needs
   a public repo or a linked private one).
2. On [share.streamlit.io](https://share.streamlit.io), "New app" → select the
   repo, branch `main`, main file `app.py`.
3. No secrets are required — the app runs fully offline (no API keys).
4. First load will build the environment from `requirements.txt` (a few
   minutes, since `faster-whisper`/`resemblyzer` pull in ctranslate2/torch);
   subsequent loads are fast.

**Option B — Hugging Face Spaces (Docker):** the repo includes a `Dockerfile`
(listens on port 7860, the port HF's Docker SDK expects). Create a Space at
[huggingface.co/new-space](https://huggingface.co/new-space) with SDK
**Docker**, then push this repo to the Space's git remote. HF Spaces' free
CPU-basic tier is more generous on RAM (~16GB) than Streamlit Cloud's (~1GB),
which matters for the `torch`/`faster-whisper` stack — but as of writing,
non-Static Spaces may require a verified payment method on the account even
on the free tier.

**Option C — self-hosted (e.g. AWS EC2 free tier):** `streamlit run app.py
--server.port=8501 --server.address=0.0.0.0` behind any reverse proxy, or run
the included `Dockerfile` directly. A free-tier `t2.micro`/`t3.micro` only has
1GB RAM, so add a swap file before `pip install -r requirements.txt`.

## 9. Known limitations

- Teacher/Student split is a 2-way approximation, not per-student identity.
- Question/response detection is heuristic (keyword + timing window), not
  semantic — sarcastic statements ending in "?" count as questions; a student
  answering out of the 12s window isn't credited.
- Silence includes non-speech classroom noise picked up by Whisper's VAD as
  "no speech", which isn't always literal silence.
- Whisper's language auto-detect can pick the wrong language on very short or
  noisy clips; pin it with `--language hi` (or another ISO code) when known.
- In streaming mode, early partial results can be noisier than the final one
  (role assignment and metrics refine as more of the recording is seen —
  see [§2a](#2a-streaming-design)).
